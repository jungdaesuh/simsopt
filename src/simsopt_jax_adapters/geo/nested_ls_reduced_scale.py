"""Build the archived 255×64 nested-LS problem from the F3 frozen bundle.

Gate 1 of the reduced nested-LS track. Coils come from the bundle's native
Biot-Savart JSON; the surface comes from the flat-675 runtime spec. This
does not evaluate F3's fused objective and does not inherit the 7.70× claim.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import simsopt
import simsoptpp
from jax.experimental import io_callback
from jax.scipy.sparse.linalg import gmres as jax_gmres
from numpy.typing import NDArray
from simsopt._core.json import GSONDecoder
from simsopt.field import BiotSavart
from simsopt.geo import BoozerSurface, SurfaceXYZTensorFourier, Volume
from simsopt_jax.geo.optimizers.optimizer import dense_operator_chunk_batch_size

from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.flat675.bundle import load_flat675_bundle
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_CONSTRAINT_WEIGHT,
    NESTED_LS_NEWTON_MAXITER,
    NESTED_LS_NEWTON_STAB,
    NESTED_LS_NEWTON_TOL,
    NESTED_LS_WEIGHT_INV_MODB,
    nested_ls_banana_run_code_options,
    nested_ls_physics_newton_kwargs,
)
from simsopt_jax_adapters.geo.nested_ls_reduced import (
    NESTED_LS_IMPLICIT_ADJOINT_DEFAULT_DENSE_BYTES,
    NESTED_LS_SCHUR_GMRES_MAXITER_CAP,
    NESTED_LS_SCHUR_GMRES_RESTART,
    NESTED_LS_SCHUR_GMRES_RTOL,
    NestedLsB37TimingBlocked,
    NestedLsReducedRankError,
    NestedLsSchurNewtonStepRecord,
    apply_reduced_mixed_schur_coil_tangent,
    compare_ad_qr_and_schur_hvp,
    dense_schur_inverse_preconditioner,
    eisenstat_walker_forcing_eta,
    factor_reduced_nested_ls_schur,
    factor_schur_fourier_block_preconditioner,
    implicit_adjoint_coil_gradient,
    linear_solve_meets_forcing,
    materialize_stabilized_schur_dense,
    nested_ls_reduced_closures,
    nested_ls_runtime_coil_closures,
    pack_surface_and_y,
    reduced_penalty_gradient,
    reduced_penalty_gradient_envelope,
    reduced_penalty_hvp,
    require_full_y_rank,
    run_reduced_nested_ls_newton,
    run_reduced_nested_ls_schur_newton,
    schur_dense_operator_bytes,
    solve_operator_gmres_with_forcing,
    solve_projected_y,
    solve_stabilized_schur_dense_lu,
)

KIB_PER_GIB = 1024 * 1024

DEFAULT_FLAT675_BUNDLE_ROOT = (
    Path.home() / "simsopt_mixed_artifacts" / "genuine675-r3-input-1c23f6c5-20260721-r1"
)
NATIVE_BIOT_SAVART_FILENAME = "native_biot_savart.json"
DEFAULT_F3_B37_RUN = (
    Path.home()
    / "simsopt_mixed_artifacts"
    / "flat675_fused_campaign"
    / "20260819T163816Z-pairs-b37-2751085"
)
DEFAULT_F3_B37_GPU_LANE = DEFAULT_F3_B37_RUN / "pair2-l1" / "lane.json"
DEFAULT_F3_B37_NATIVE_LANE = DEFAULT_F3_B37_RUN / "pair2-l2" / "lane.json"

# Reconstruct §2.1 archived-start QR inner state (C++ Newton was a no-op).
ARCHIVED_START_QR_IOTA = 0.1500517839808274
ARCHIVED_START_QR_G = 2.010619295609829
F3_B37_STEP3_SURFACE_SHA256 = (
    "286e3dabf3c9113d25aa691e1abe36a109057738e67536d9e46e6d56faa17e24"
)
# Walk step 4 requested η (Choice 2). Distinct from achieved η=0.1203881060498997.
F3_B37_STEP4_ETA_REQUESTED = 0.04071795165373735
F3_B37_STEP4_CAP64_ETA_ACHIEVED = 0.1203881060498997
F3_B37_STEP4_CAP64_RESIDUAL_L2 = 9.402277895763763e-05
F3_B37_STEP5_SURFACE_SHA256 = (
    "a0493560d7ebe3455b68bb834830ad59fb1fb510f79a447c61267547cdc0effe"
)
F3_B37_STEP6_IOTA = 0.1484103489869863
F3_B37_STEP6_G = 2.0106193052280394
F3_B37_STEP6_GRAD_L2 = 0.0001449305895138173
F3_B37_STEP6_ETA_REQUESTED = 0.027034810094191494
F3_B37_STEP6_CAP512_ETA_ACHIEVED = 0.09404256261986046
F3_B37_STEP6_MATERIAL_RESIDUAL_RATIO = 0.9
F3_B37_STEP6_WALL_SECONDS_2048 = 1200.0
F3_B37_STEP6_CAP_2048 = 2048
F3_B37_STEP6_CAP_2048_START_MAXITER = 2048
F3_B37_STEP6_ARCH_RESTARTS = (8, 32, 64, 128)
F3_B37_STEP6_ARCH_HVP_BUDGET = 128
F3_B37_STEP6_ARCH_SOLVE_METHODS = ("incremental", "batched")
F3_B37_STEP6_ARCH_SWEEP_JAX_TOL = 1.0e-16
F3_B37_STEP6_ARCH_WALL_SECONDS = 900.0
F3_B37_STEP6_ARCH_FULL_RESTART = 661
F3_B37_CAP512_WALK_EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "receipts"
    / "evidence"
    / "nested_ls_reduced_gpu_walk_20260821.cap512.incomplete.json"
)
F3_B37_DENSE_LU_WALK_EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "receipts"
    / "evidence"
    / "nested_ls_reduced_gpu_walk_20260821.dense_lu.json"
)
F3_B37_DENSE_LU_ENDPOINT_SURFACE_SHA256 = (
    "e25ca0f2fedf25cf411f7b7ad7192860c813ad5fd37bdb5471355834d42ede6c"
)
F3_B37_DENSE_LU_ENDPOINT_IOTA = 0.14085710957665173
F3_B37_DENSE_LU_ENDPOINT_G = 2.0106193053897154
F3_B37_DENSE_LU_ENDPOINT_GRAD_L2 = 2.404212353322172e-14
F3_B37_IFT_STAB = 0.0
F3_B37_ADJOINT_COIL_SCAN = 8
F3_B37_ADJOINT_FD_EPSILON = 1.0e-6
F3_B37_ADJOINT_WALL_SECONDS = 1800.0
F3_B37_ADJOINT_LIVE_ETA_TOL = 1.0e-10
F3_B37_ADJOINT_VJP_RTOL = 1.0e-6
F3_B37_ADJOINT_VJP_ATOL = 1.0e-8
F3_B37_ADJOINT_FD_RTOL = 5.0e-2
F3_B37_ADJOINT_FD_ATOL_FLOOR = 1.0e-9
F3_B37_ADJOINT_FD_ATOL_REL = 1.0e-2
F3_B37_ADJOINT_MIXED_NORM_FLOOR = 1.0e-8
F3_B37_CHUNK_WIDTHS = (8, 16, 32, 64)
F3_B37_CHUNK_BANANA_WALL_SECONDS = 1800.0
F3_B37_VOLUME_OUTER_WALL_SECONDS = 1800.0
F3_B37_BANANA_OMP_THREADS = (4, 8, 16, 32)
F3_B37_BANANA_OMP_REPEATS = 2
F3_B37_CHUNK_WARM_REPEATS = 3
F3_B37_BANANA_OMP_WALL_SECONDS = 3600.0


def archived_flat675_bundle_available(bundle_root: Path | None = None) -> bool:
    """True when the host-local F3 bundle and native Biot-Savart JSON exist."""

    root = DEFAULT_FLAT675_BUNDLE_ROOT if bundle_root is None else Path(bundle_root)
    return root.is_dir() and (root / NATIVE_BIOT_SAVART_FILENAME).is_file()


def archived_f3_b37_lanes_available() -> bool:
    """True when the host-local fused and native B37 lane JSON files exist."""

    return DEFAULT_F3_B37_GPU_LANE.is_file() and DEFAULT_F3_B37_NATIVE_LANE.is_file()


def _load_native_biot_savart(bundle_root: Path) -> BiotSavart:
    biot_savart = json.loads(
        (bundle_root / NATIVE_BIOT_SAVART_FILENAME).read_text(),
        cls=GSONDecoder,
    )
    if not isinstance(biot_savart, BiotSavart):
        raise TypeError(
            "archived native_biot_savart.json must decode to BiotSavart; "
            f"got {type(biot_savart).__name__}."
        )
    return biot_savart


def load_flat675_lane_blocks(
    lane_path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, object]]:
    """Return coil and surface blocks from a fused or native F3 lane JSON."""

    payload = json.loads(Path(lane_path).read_text())
    result = payload["result"]
    endpoint = result.get("endpoint_candidate")
    if isinstance(endpoint, dict) and "coil_coordinates" in endpoint:
        candidate = endpoint
        schema = "fused_endpoint_candidate"
        inner = result.get("endpoint_inner_state")
    else:
        certificate = result["final_certificate"]
        candidate = certificate["candidate"]
        schema = "native_final_certificate"
        inner = certificate["y_certificate"]["solution"]
    coils = np.asarray(candidate["coil_coordinates"], dtype=np.float64)
    surface = np.asarray(candidate["surface_coordinates"], dtype=np.float64)
    if coils.ndim != 1 or surface.ndim != 1:
        raise ValueError("lane coil and surface blocks must be 1-D float64 vectors.")
    return (
        coils,
        surface,
        {
            "schema": schema,
            "path": str(Path(lane_path)),
            "lane": payload.get("lane") or result.get("lane"),
            "inner_state": (
                [float(inner[0]), float(inner[1])] if inner is not None else None
            ),
        },
    )


def _clone_tensor_surface(surface: SurfaceXYZTensorFourier) -> SurfaceXYZTensorFourier:
    cloned = SurfaceXYZTensorFourier(
        mpol=int(surface.mpol),
        ntor=int(surface.ntor),
        nfp=int(surface.nfp),
        stellsym=bool(surface.stellsym),
        clamped_dims=list(surface.clamped_dims),
        quadpoints_phi=np.asarray(surface.quadpoints_phi, dtype=np.float64).copy(),
        quadpoints_theta=np.asarray(surface.quadpoints_theta, dtype=np.float64).copy(),
    )
    cloned.set_dofs(np.asarray(surface.get_dofs(), dtype=np.float64).copy())
    return cloned


def _surface_from_template(template, surface_coordinates: NDArray[np.float64]):
    surface = SurfaceXYZTensorFourier(
        mpol=int(template.mpol),
        ntor=int(template.ntor),
        nfp=int(template.nfp),
        stellsym=bool(template.stellsym),
        clamped_dims=list(template.clamped_dims),
        quadpoints_phi=np.asarray(template.quadpoints_phi, dtype=np.float64),
        quadpoints_theta=np.asarray(template.quadpoints_theta, dtype=np.float64),
    )
    surface.set_dofs(np.asarray(surface_coordinates, dtype=np.float64))
    return surface


def load_archived_nested_ls_pair(
    bundle_root: Path | None = None,
    *,
    coil_coordinates: object | None = None,
    surface_coordinates: object | None = None,
    materialize_dense_linearization: bool = False,
) -> tuple[BoozerSurface, BoozerSurfaceJAX, float]:
    """Native C++ and JAX LS surfaces on frozen archived coils.

    Default coordinates are the bundle start. Pass the F3 B37 coil and
    surface blocks to evaluate that frozen-coil point instead.

    Dense Hessian materialization is off: assembling ``H_ss`` at 661
    through autodiff-of-QR is a Gate 3 measurement, not a Gate 1
    stationarity check.
    """

    root = DEFAULT_FLAT675_BUNDLE_ROOT if bundle_root is None else Path(bundle_root)
    problem = load_flat675_bundle(root)
    native_field = _load_native_biot_savart(root)
    jax_field = _load_native_biot_savart(root)
    template = problem.material.boozer.surface_template
    start = problem.start_candidate
    coils = (
        np.asarray(start.coil_coordinates, dtype=np.float64)
        if coil_coordinates is None
        else np.asarray(coil_coordinates, dtype=np.float64)
    )
    surface_dofs = (
        np.asarray(start.surface_coordinates, dtype=np.float64)
        if surface_coordinates is None
        else np.asarray(surface_coordinates, dtype=np.float64)
    )
    archived_x = np.asarray(native_field.x, dtype=np.float64)
    if coils.shape != archived_x.shape:
        raise ValueError(
            "coil coordinates shape "
            f"{coils.shape} does not match archived BiotSavart.x "
            f"{archived_x.shape}."
        )
    native_field.x = coils
    jax_field.x = np.array(coils, dtype=np.float64, copy=True)
    surface = _surface_from_template(template, surface_dofs)
    native_surface = _clone_tensor_surface(surface)
    jax_surface = _clone_tensor_surface(surface)
    target = float(problem.objective_policy.boozer_target_label)
    newton_options = {
        "verbose": False,
        "newton_tol": NESTED_LS_NEWTON_TOL,
        "newton_maxiter": NESTED_LS_NEWTON_MAXITER,
        "weight_inv_modB": NESTED_LS_WEIGHT_INV_MODB,
    }
    native = BoozerSurface(
        native_field,
        native_surface,
        Volume(native_surface),
        target,
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        options=newton_options,
    )
    jax_boozer = BoozerSurfaceJAX(
        BiotSavartJAX(jax_field.coils),
        jax_surface,
        Volume(jax_surface),
        target,
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        options={
            **newton_options,
            "optimizer_backend": "scipy",
            "materialize_dense_linearization": bool(materialize_dense_linearization),
        },
    )
    return native, jax_boozer, target


def kib_to_gib(kib: int) -> float:
    """Convert Linux kibibytes to gibibytes (1024³ bytes)."""

    return float(kib) / float(KIB_PER_GIB)


def float64_ulps(value: float, reference: float) -> float:
    """Count float64 ULPs of ``value`` from ``reference`` using ``np.spacing``."""

    left = np.float64(value)
    right = np.float64(reference)
    spacing = np.spacing(left)
    return float(abs(left - right) / spacing)


def nested_ls_threading_env() -> dict[str, str | None]:
    """OpenMP and BLAS thread knobs observed in this process."""

    return {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OMP_PROC_BIND": os.environ.get("OMP_PROC_BIND"),
        "OMP_PLACES": os.environ.get("OMP_PLACES"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
    }


def nested_ls_omp_threads_pinned(env: dict[str, str | None] | None = None) -> bool:
    """True when ``OMP_NUM_THREADS`` is a positive integer in the process env."""

    observed = nested_ls_threading_env() if env is None else env
    raw = observed.get("OMP_NUM_THREADS")
    if raw is None or str(raw).strip() == "":
        return False
    try:
        return int(str(raw).strip()) >= 1
    except ValueError:
        return False


def nested_ls_runtime_identity() -> dict[str, object]:
    """Host and JAX device identity for an evidence document."""

    threading = nested_ls_threading_env()
    return {
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
        "jax_enable_x64_env": os.environ.get("JAX_ENABLE_X64"),
        "threading": threading,
        "omp_num_threads": threading["OMP_NUM_THREADS"],
        "omp_pinned": nested_ls_omp_threads_pinned(threading),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def dump_strict_json(payload: dict[str, object]) -> str:
    """Serialize evidence with ``allow_nan=False`` so NaN cannot sneak in."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _as_int(value: object) -> int:
    """JSON number as int. Bool is not a JSON number."""

    if isinstance(value, bool):
        raise TypeError("bool is not a JSON number")
    return int(cast(int | float, value))


def _as_float(value: object) -> float:
    """JSON number as float. Bool is not a JSON number."""

    if isinstance(value, bool):
        raise TypeError("bool is not a JSON number")
    return float(cast(int | float, value))


def last_step_meets_forcing(steps: Sequence[object]) -> bool:
    """True when the last recorded step's η_achieved is at most its η_k.

    Rejected last steps count. Do not use only accepted steps or the
    walk-level ``gmres_rtol`` cap.
    """

    records = tuple(steps)
    if not records:
        return False
    last = records[-1]
    if isinstance(last, dict):
        eta = _as_float(last["gmres_forcing_eta"])
        requested = _as_float(last["gmres_rtol"])
    else:
        step = cast(NestedLsSchurNewtonStepRecord, last)
        eta = _as_float(step.gmres_forcing_eta)
        requested = _as_float(step.gmres_rtol)
    return linear_solve_meets_forcing(eta, requested)


def gmres_doubling_cycle_budget(start_maxiter: int, cap: int) -> int:
    """Sum of JAX GMRES ``maxiter`` values along the doubling schedule.

    ``solve_operator_gmres_with_forcing`` starts at ``start_maxiter``
    and doubles until ``cap``. Starting below the cap therefore
    re-pays every previous budget (1+2+…+cap, or start+2·start+…+cap).
    """

    used = max(1, int(start_maxiter))
    cap_i = max(int(cap), used)
    total = 0
    while True:
        total += used
        if used >= cap_i:
            return int(total)
        used = min(cap_i, used * 2)


def predict_start_at_cap_wall_seconds(
    previous_seconds: float,
    *,
    previous_start_maxiter: int,
    previous_cap: int,
    next_cap: int,
) -> float:
    """Predicted wall when the next leg starts at ``next_cap`` cycles.

    Scales the previous row by ``next_cap / previous_doubling_budget``
    so the predictor models a start-at-cap leg, not a double-pay.
    """

    previous_cycles = gmres_doubling_cycle_budget(previous_start_maxiter, previous_cap)
    if (
        previous_cycles <= 0
        or not np.isfinite(previous_seconds)
        or float(previous_seconds) < 0.0
    ):
        return float("inf")
    return float(previous_seconds) * (float(next_cap) / float(previous_cycles))


def unpreconditioned_gmres_is_insufficient(
    fail_reason: str | None,
    *,
    residual_ratio: float | None = None,
    material_residual_ratio: float = F3_B37_STEP6_MATERIAL_RESIDUAL_RATIO,
) -> bool:
    """True only for stagnation, or η-unmet with a failed residual test.

    Budget exhaustion while the residual is still falling is not
    insufficiency. ``residual_ratio`` is current/first residual; the
    material test fails when that ratio is at least
    ``material_residual_ratio``.
    """

    if fail_reason == "stagnation":
        return True
    if fail_reason != "eta_unmet":
        return False
    if residual_ratio is None or not np.isfinite(residual_ratio):
        return True
    return float(residual_ratio) >= float(material_residual_ratio)


class NestedLsCountedMatvec:
    """Count operator applications through JAX GMRES control flow."""

    __slots__ = ("_matvec", "count")

    def __init__(self, matvec: Callable[[jax.Array], jax.Array]) -> None:
        self._matvec = matvec
        self.count = 0

    def __call__(self, tangent: jax.Array) -> jax.Array:
        def _bump(_: object) -> None:
            self.count += 1

        io_callback(
            _bump,
            None,
            jnp.asarray(0, dtype=jnp.int32),
            ordered=True,
        )
        return self._matvec(tangent)


def write_strict_json(path: Path, payload: dict[str, object]) -> None:
    """Author an evidence JSON file. Pytest must not call this."""

    path.write_text(dump_strict_json(payload), encoding="utf-8")


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's raw bytes."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_float64(values: object) -> str:
    """SHA-256 of a C-contiguous float64 buffer."""

    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest()


def _git_output(*args: str) -> str:
    repo = Path(__file__).resolve().parents[3]
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def nested_ls_receipt_provenance() -> dict[str, object]:
    """Commit, versions, source hashes, and F3 input hashes for a receipt."""

    repo = Path(__file__).resolve().parents[3]
    dirty = _git_output("status", "--porcelain")
    biot_path = DEFAULT_FLAT675_BUNDLE_ROOT / NATIVE_BIOT_SAVART_FILENAME
    simsoptpp_file = simsoptpp.__file__
    if simsoptpp_file is None:
        raise RuntimeError("simsoptpp has no __file__; cannot hash the extension.")
    simsoptpp_path = Path(simsoptpp_file).resolve()
    return {
        **nested_ls_runtime_identity(),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(dirty),
        "git_status_porcelain": dirty,
        "python_version": sys.version,
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "simsopt_version": simsopt.__version__,
        "simsoptpp_path": str(simsoptpp_path),
        "simsoptpp_sha256": sha256_file(simsoptpp_path),
        "source_sha256": {
            "nested_ls_contract.py": sha256_file(
                repo / "src/simsopt_jax_adapters/geo/nested_ls_contract.py"
            ),
            "nested_ls_reduced.py": sha256_file(
                repo / "src/simsopt_jax_adapters/geo/nested_ls_reduced.py"
            ),
            "nested_ls_reduced_scale.py": sha256_file(
                repo / "src/simsopt_jax_adapters/geo/nested_ls_reduced_scale.py"
            ),
        },
        "input_sha256": {
            "native_biot_savart.json": (
                sha256_file(biot_path) if biot_path.is_file() else None
            ),
            "pair2-l1_lane.json": (
                sha256_file(DEFAULT_F3_B37_GPU_LANE)
                if DEFAULT_F3_B37_GPU_LANE.is_file()
                else None
            ),
        },
    }


def _peak_rss_kib() -> int:
    """Linux ``ru_maxrss`` is kibibytes and is the process-lifetime peak."""

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _current_rss_kib() -> int:
    with Path("/proc/self/status").open() as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return _peak_rss_kib()


@dataclass(frozen=True, slots=True)
class NestedLsBoundedF3B37Result:
    """Independent C++ reference plus a bounded JAX HVP / one-step probe.

    This is not a ten-step reduced walk and not a nested timing claim.
    Trajectories need not match; the native rejudge certifies the branch.
    """

    residual_rows: int
    y_star_iota: float
    y_star_g: float
    y_rank: int
    reduced_grad_l2: float
    reduced_grad_finite: bool
    hvp_l2: float
    hvp_finite: bool
    hvp_seconds: float
    rss_kib_after_hvp: int
    peak_rss_kib_after_hvp: int
    one_step_iter: int
    one_step_success: bool
    one_step_finite: bool
    one_step_iota: float
    one_step_g: float
    one_step_grad_l2: float
    one_step_seconds: float
    rss_kib_after_one_step: int
    peak_rss_kib_after_one_step: int
    one_step_coil_delta_inf: float
    one_step_attempted: bool
    native_ref_success: bool
    native_ref_iter: int
    native_ref_iota: float
    native_ref_g: float
    native_ref_delta_iota: float
    native_ref_delta_surface_inf: float
    native_ref_coil_delta_inf: float
    native_ref_seconds: float
    native_rejudge_success: bool
    native_rejudge_iter: int
    native_rejudge_iota: float | None
    native_rejudge_g: float | None
    native_rejudge_coil_delta_inf: float
    native_rejudge_seconds: float
    full_walk_attempted: bool
    schur_hvp_l2: float | None
    schur_hvp_finite: bool | None
    schur_hvp_seconds: float | None
    schur_factor_seconds: float | None
    schur_vs_ad_max_abs: float | None
    schur_vs_ad_rel_l2: float | None
    schur_phi_yy_condition: float | None

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def evaluate_f3_b37_bounded_probe(
    native: BoozerSurface,
    jax_boozer: BoozerSurfaceJAX,
    *,
    one_newton_step: bool = False,
    compare_schur_hvp: bool = False,
) -> NestedLsBoundedF3B37Result:
    """C++ full reference plus JAX ``y*`` / ``g`` / one HVP.

    The JAX lane starts from the original F3 surface, not from the C++
    endpoint. One Newton step is opt-in: autodiff-through-QR GMRES is a
    many-HVP solve and is not the default bounded gate. Does not launch
    a ten-step reduced walk. Unattempted rejudge scalars are ``None``.
    """

    residual_fn, _objective_fn, phi_hat = nested_ls_reduced_closures(jax_boozer)
    jax_surface0 = np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64)
    native_surface0 = np.asarray(native.surface.get_dofs(), dtype=np.float64)
    native_coils0 = np.asarray(native.biotsavart.x, dtype=np.float64)
    solution = solve_projected_y(
        residual_fn, jax_surface0, np.zeros(2, dtype=np.float64)
    )
    require_full_y_rank(solution)
    y_star = np.asarray(solution.solution, dtype=np.float64)
    residual_rows = int(np.asarray(solution.design_matrix).shape[0])
    packed_zero = pack_surface_and_y(jax_surface0, np.zeros(2, dtype=np.float64))
    if int(np.asarray(residual_fn(packed_zero)).shape[0]) != residual_rows:
        raise ValueError("Gate 1 residual length disagrees with the QR design matrix.")

    gradient = np.asarray(
        reduced_penalty_gradient(phi_hat, jax_surface0), dtype=np.float64
    )
    grad_l2 = float(np.linalg.norm(gradient))
    tangent = (
        gradient / grad_l2
        if np.isfinite(grad_l2) and grad_l2 > 0.0
        else (np.zeros_like(gradient))
    )
    hvp_started = time.perf_counter()
    hvp = np.asarray(
        reduced_penalty_hvp(phi_hat, jax_surface0, tangent), dtype=np.float64
    )
    hvp_seconds = time.perf_counter() - hvp_started
    rss_after_hvp = _current_rss_kib()
    peak_after_hvp = _peak_rss_kib()

    if compare_schur_hvp:
        factor_started = time.perf_counter()
        schur_operator = factor_reduced_nested_ls_schur(
            residual_fn, _objective_fn, jax_surface0
        )
        schur_factor_seconds = time.perf_counter() - factor_started
        schur_started = time.perf_counter()
        schur_hvp = np.asarray(schur_operator.apply(tangent), dtype=np.float64)
        schur_hvp_seconds = time.perf_counter() - schur_started
        schur_delta = schur_hvp - hvp
        ad_norm = float(np.linalg.norm(hvp))
        schur_hvp_l2 = float(np.linalg.norm(schur_hvp))
        schur_hvp_finite = bool(np.all(np.isfinite(schur_hvp)))
        schur_vs_ad_max_abs = float(np.max(np.abs(schur_delta)))
        schur_vs_ad_rel_l2 = (
            float(np.linalg.norm(schur_delta) / ad_norm)
            if ad_norm > 0.0
            else float(np.linalg.norm(schur_delta))
        )
        schur_phi_yy_condition = float(schur_operator.phi_yy_condition)
    else:
        schur_hvp_l2 = None
        schur_hvp_finite = None
        schur_hvp_seconds = None
        schur_factor_seconds = None
        schur_vs_ad_max_abs = None
        schur_vs_ad_rel_l2 = None
        schur_phi_yy_condition = None

    native.need_to_run_code = True
    native_started = time.perf_counter()
    native_ref = native.minimize_boozer_penalty_constraints_newton(
        iota=float(y_star[0]),
        G=float(y_star[1]),
        **nested_ls_physics_newton_kwargs(),
    )
    native_ref_seconds = time.perf_counter() - native_started
    native_surface_ref = np.asarray(native.surface.get_dofs(), dtype=np.float64)
    native_coils_ref = np.asarray(native.biotsavart.x, dtype=np.float64)
    native_ref_iota = float(native_ref["iota"])
    native_ref_g = float(native_ref["G"])

    if one_newton_step:
        one_step_started = time.perf_counter()
        one_step = run_reduced_nested_ls_newton(
            jax_boozer,
            iota=float(y_star[0]),
            G=float(y_star[1]),
            maxiter=1,
        )
        one_step_seconds = time.perf_counter() - one_step_started
        rss_after_one_step = _current_rss_kib()
        peak_after_one_step = _peak_rss_kib()
        one_step_finite = bool(
            np.all(np.isfinite(one_step.surface_dofs))
            and np.all(np.isfinite(one_step.reduced_gradient))
            and np.isfinite(one_step.iota)
            and np.isfinite(one_step.G)
        )
        native.need_to_run_code = True
        native.surface.set_dofs(one_step.surface_dofs)
        rejudge_coils0 = np.asarray(native.biotsavart.x, dtype=np.float64)
        rejudge_started = time.perf_counter()
        native_rejudge = native.minimize_boozer_penalty_constraints_newton(
            iota=float(one_step.iota),
            G=float(one_step.G),
            **nested_ls_physics_newton_kwargs(),
        )
        rejudge_seconds = time.perf_counter() - rejudge_started
        rejudge_coils1 = np.asarray(native.biotsavart.x, dtype=np.float64)
        one_step_iter = int(one_step.iteration_count)
        one_step_success = bool(one_step.success)
        one_step_iota = float(one_step.iota)
        one_step_g = float(one_step.G)
        one_step_grad_l2 = float(np.linalg.norm(one_step.reduced_gradient))
        one_step_coil = float(one_step.coil_delta_inf)
        rejudge_success = bool(native_rejudge["success"])
        rejudge_iter = int(native_rejudge["iter"])
        rejudge_iota = float(native_rejudge["iota"])
        rejudge_g = float(native_rejudge["G"])
        rejudge_coil = float(
            np.linalg.norm(rejudge_coils1 - rejudge_coils0, ord=np.inf)
        )
    else:
        one_step_seconds = 0.0
        rss_after_one_step = rss_after_hvp
        peak_after_one_step = peak_after_hvp
        one_step_finite = False
        one_step_iter = 0
        one_step_success = False
        one_step_iota = float(y_star[0])
        one_step_g = float(y_star[1])
        one_step_grad_l2 = grad_l2
        one_step_coil = 0.0
        rejudge_success = False
        rejudge_iter = 0
        rejudge_iota = None
        rejudge_g = None
        rejudge_coil = 0.0
        rejudge_seconds = 0.0
    return NestedLsBoundedF3B37Result(
        residual_rows=residual_rows,
        y_star_iota=float(y_star[0]),
        y_star_g=float(y_star[1]),
        y_rank=int(np.asarray(solution.numerical_rank)),
        reduced_grad_l2=grad_l2,
        reduced_grad_finite=bool(np.all(np.isfinite(gradient))),
        hvp_l2=float(np.linalg.norm(hvp)),
        hvp_finite=bool(np.all(np.isfinite(hvp))),
        hvp_seconds=float(hvp_seconds),
        rss_kib_after_hvp=int(rss_after_hvp),
        peak_rss_kib_after_hvp=int(peak_after_hvp),
        one_step_iter=one_step_iter,
        one_step_success=one_step_success,
        one_step_finite=one_step_finite,
        one_step_iota=one_step_iota,
        one_step_g=one_step_g,
        one_step_grad_l2=one_step_grad_l2,
        one_step_seconds=float(one_step_seconds),
        rss_kib_after_one_step=int(rss_after_one_step),
        peak_rss_kib_after_one_step=int(peak_after_one_step),
        one_step_coil_delta_inf=one_step_coil,
        one_step_attempted=bool(one_newton_step),
        native_ref_success=bool(native_ref["success"]),
        native_ref_iter=int(native_ref["iter"]),
        native_ref_iota=native_ref_iota,
        native_ref_g=native_ref_g,
        native_ref_delta_iota=native_ref_iota - float(y_star[0]),
        native_ref_delta_surface_inf=float(
            np.linalg.norm(native_surface_ref - native_surface0, ord=np.inf)
        ),
        native_ref_coil_delta_inf=float(
            np.linalg.norm(native_coils_ref - native_coils0, ord=np.inf)
        ),
        native_ref_seconds=float(native_ref_seconds),
        native_rejudge_success=rejudge_success,
        native_rejudge_iter=rejudge_iter,
        native_rejudge_iota=rejudge_iota,
        native_rejudge_g=rejudge_g,
        native_rejudge_coil_delta_inf=rejudge_coil,
        native_rejudge_seconds=float(rejudge_seconds),
        full_walk_attempted=False,
        schur_hvp_l2=schur_hvp_l2,
        schur_hvp_finite=schur_hvp_finite,
        schur_hvp_seconds=schur_hvp_seconds,
        schur_factor_seconds=schur_factor_seconds,
        schur_vs_ad_max_abs=schur_vs_ad_max_abs,
        schur_vs_ad_rel_l2=schur_vs_ad_rel_l2,
        schur_phi_yy_condition=schur_phi_yy_condition,
    )


@dataclass(frozen=True, slots=True)
class NestedLsSchurNewtonStepProbe:
    """One capped Schur Newton step plus a C++ rejudge of that JAX state."""

    y_star_iota: float
    y_star_g: float
    reduced_grad_l2_before: float
    step_iter: int
    step_success: bool
    step_persisted: bool
    step_accepted: bool
    step_alpha: float
    step_iota: float
    step_g: float
    step_grad_l2: float
    step_coil_delta_inf: float
    step_seconds: float
    gmres_info: int
    gmres_matvecs: int
    gmres_residual_l2: float
    gmres_forcing_eta: float
    gmres_rtol: float
    gmres_restart: int
    gmres_maxiter: int
    factor_seconds: float
    gmres_seconds: float
    phi_yy_condition: float
    rss_kib_after_step: int
    peak_rss_kib_after_step: int
    native_rejudge_success: bool
    native_rejudge_iter: int
    native_rejudge_iota: float
    native_rejudge_g: float
    native_rejudge_coil_delta_inf: float
    native_rejudge_seconds: float
    native_rejudge_grad_l2: float
    native_rejudge_grad_inf: float
    native_rejudge_surface_sha256: str
    reconstruct_ref_iota: float | None
    reconstruct_ref_g: float | None
    reconstruct_ref_grad_l2: float | None
    reconstruct_ref_surface_sha256: str | None
    rejudge_vs_reconstruct_surface_inf: float | None
    schur_vs_ad_rel_l2: float | None
    schur_vs_ad_max_abs: float | None
    phi_yy_condition_from_hvp: float | None
    runtime: dict[str, object]
    provenance: dict[str, object]

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def _native_ls_gradient_norms(result) -> tuple[float, float]:
    gradient = np.asarray(result["jacobian"], dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(gradient)), float(np.linalg.norm(gradient, ord=np.inf))


def evaluate_f3_b37_schur_newton_step(
    native: BoozerSurface,
    jax_boozer: BoozerSurfaceJAX,
    *,
    compare_reconstruct_branch: bool = True,
    compare_schur_hvp: bool = True,
) -> NestedLsSchurNewtonStepProbe:
    """One capped Schur Newton step from the F3 B37 point, then C++ rejudge.

    Does not run a ten-step walk and does not time nested-LS versus F3.
    The live Krylov is JAX incremental GMRES; this is not a GPU timing claim.
    """

    residual_fn, objective_fn, phi_hat = nested_ls_reduced_closures(jax_boozer)
    surface0 = np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64)
    native_start = np.asarray(native.surface.get_dofs(), dtype=np.float64)
    solution = solve_projected_y(residual_fn, surface0, np.zeros(2, dtype=np.float64))
    require_full_y_rank(solution)
    y_star = np.asarray(solution.solution, dtype=np.float64)
    gradient = np.asarray(
        reduced_penalty_gradient_envelope(objective_fn, surface0, y_star),
        dtype=np.float64,
    )
    grad_l2 = float(np.linalg.norm(gradient))
    tangent = (
        gradient / grad_l2
        if np.isfinite(grad_l2) and grad_l2 > 0.0
        else np.zeros_like(gradient)
    )
    if compare_schur_hvp:
        comparison = compare_ad_qr_and_schur_hvp(
            residual_fn, objective_fn, phi_hat, surface0, tangent
        )
        schur_vs_ad_rel_l2 = float(comparison.rel_l2)
        schur_vs_ad_max_abs = float(comparison.max_abs)
        phi_yy_condition_from_hvp = float(comparison.phi_yy_condition)
    else:
        schur_vs_ad_rel_l2 = None
        schur_vs_ad_max_abs = None
        phi_yy_condition_from_hvp = None
    provenance = nested_ls_receipt_provenance()
    runtime = {
        "hostname": provenance["hostname"],
        "python_executable": provenance["python_executable"],
        "jax_default_backend": provenance["jax_default_backend"],
        "jax_devices": provenance["jax_devices"],
        "jax_platforms_env": provenance["jax_platforms_env"],
        "jax_enable_x64_env": provenance["jax_enable_x64_env"],
        "timestamp_utc": provenance["timestamp_utc"],
    }
    step_started = time.perf_counter()
    step = run_reduced_nested_ls_schur_newton(
        jax_boozer,
        iota=float(y_star[0]),
        G=float(y_star[1]),
        maxiter=1,
    )
    step_seconds = time.perf_counter() - step_started
    rss_after_step = _current_rss_kib()
    peak_after_step = _peak_rss_kib()
    newton_kwargs = nested_ls_physics_newton_kwargs()
    if compare_reconstruct_branch:
        native.need_to_run_code = True
        native.surface.set_dofs(native_start)
        reconstruct_ref = native.minimize_boozer_penalty_constraints_newton(
            iota=float(y_star[0]),
            G=float(y_star[1]),
            **newton_kwargs,
        )
        reconstruct_surface = np.asarray(native.surface.get_dofs(), dtype=np.float64)
        reconstruct_ref_iota = float(reconstruct_ref["iota"])
        reconstruct_ref_g = float(reconstruct_ref["G"])
        reconstruct_ref_grad_l2, _reconstruct_inf = _native_ls_gradient_norms(
            reconstruct_ref
        )
        del _reconstruct_inf
        reconstruct_ref_surface_sha256 = sha256_float64(reconstruct_surface)
    else:
        reconstruct_surface = None
        reconstruct_ref_iota = None
        reconstruct_ref_g = None
        reconstruct_ref_grad_l2 = None
        reconstruct_ref_surface_sha256 = None
    native.need_to_run_code = True
    native.surface.set_dofs(step.surface_dofs)
    rejudge_coils0 = np.asarray(native.biotsavart.x, dtype=np.float64)
    rejudge_started = time.perf_counter()
    native_rejudge = native.minimize_boozer_penalty_constraints_newton(
        iota=float(step.iota),
        G=float(step.G),
        **newton_kwargs,
    )
    rejudge_seconds = time.perf_counter() - rejudge_started
    rejudge_coils1 = np.asarray(native.biotsavart.x, dtype=np.float64)
    rejudge_surface = np.asarray(native.surface.get_dofs(), dtype=np.float64)
    rejudge_grad_l2, rejudge_grad_inf = _native_ls_gradient_norms(native_rejudge)
    if reconstruct_surface is None:
        rejudge_vs_reconstruct_surface_inf = None
    else:
        rejudge_vs_reconstruct_surface_inf = float(
            np.linalg.norm(rejudge_surface - reconstruct_surface, ord=np.inf)
        )
    return NestedLsSchurNewtonStepProbe(
        y_star_iota=float(y_star[0]),
        y_star_g=float(y_star[1]),
        reduced_grad_l2_before=float(np.linalg.norm(gradient)),
        step_iter=int(step.iteration_count),
        step_success=bool(step.success),
        step_persisted=bool(step.persisted),
        step_accepted=bool(step.step_accepted),
        step_alpha=float(step.step_alpha),
        step_iota=float(step.iota),
        step_g=float(step.G),
        step_grad_l2=float(np.linalg.norm(step.reduced_gradient)),
        step_coil_delta_inf=float(step.coil_delta_inf),
        step_seconds=float(step_seconds),
        gmres_info=int(step.gmres_info),
        gmres_matvecs=int(step.gmres_matvecs),
        gmres_residual_l2=float(step.gmres_residual_l2),
        gmres_forcing_eta=float(step.gmres_forcing_eta),
        gmres_rtol=float(step.gmres_rtol),
        gmres_restart=int(step.gmres_restart),
        gmres_maxiter=int(step.gmres_maxiter),
        factor_seconds=float(step.factor_seconds),
        gmres_seconds=float(step.gmres_seconds),
        phi_yy_condition=float(step.phi_yy_condition),
        rss_kib_after_step=int(rss_after_step),
        peak_rss_kib_after_step=int(peak_after_step),
        native_rejudge_success=bool(native_rejudge["success"]),
        native_rejudge_iter=int(native_rejudge["iter"]),
        native_rejudge_iota=float(native_rejudge["iota"]),
        native_rejudge_g=float(native_rejudge["G"]),
        native_rejudge_coil_delta_inf=float(
            np.linalg.norm(rejudge_coils1 - rejudge_coils0, ord=np.inf)
        ),
        native_rejudge_seconds=float(rejudge_seconds),
        native_rejudge_grad_l2=rejudge_grad_l2,
        native_rejudge_grad_inf=rejudge_grad_inf,
        native_rejudge_surface_sha256=sha256_float64(rejudge_surface),
        reconstruct_ref_iota=reconstruct_ref_iota,
        reconstruct_ref_g=reconstruct_ref_g,
        reconstruct_ref_grad_l2=reconstruct_ref_grad_l2,
        reconstruct_ref_surface_sha256=reconstruct_ref_surface_sha256,
        rejudge_vs_reconstruct_surface_inf=rejudge_vs_reconstruct_surface_inf,
        schur_vs_ad_rel_l2=schur_vs_ad_rel_l2,
        schur_vs_ad_max_abs=schur_vs_ad_max_abs,
        phi_yy_condition_from_hvp=phi_yy_condition_from_hvp,
        runtime=runtime,
        provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class NestedLsSchurNewtonWalkProbe:
    """Ten-step frozen-coil Schur walk plus C++ reconstruct rejudge.

    ``iota`` / ``G`` / ``grad_l2`` are the JAX endpoint. Rejudge deltas
    are versus that JAX state, not versus the independent reconstruct
    reference. ``gmres_matvecs`` is not on this probe: JAX incremental
    GMRES does not report operator applications.
    """

    y_star_iota: float
    y_star_g: float
    reduced_grad_l2_before: float
    maxiter: int
    linear_solver: str
    iteration_count: int
    step_accepted: bool
    success: bool
    persisted: bool
    iota: float
    G: float
    grad_l2: float
    coil_delta_inf: float
    walk_seconds: float
    gmres_forcing_eta: float
    last_step_eta_achieved: float
    last_step_eta_requested: float
    last_step_forcing_ok: bool
    steps: tuple[dict[str, object], ...]
    jax_iota: float
    jax_g: float
    jax_surface_sha256: str
    jax_surface_dofs: tuple[float, ...]
    native_rejudge_success: bool
    native_rejudge_iter: int
    native_rejudge_iota: float
    native_rejudge_g: float
    native_rejudge_coil_delta_inf: float
    native_rejudge_seconds: float
    native_rejudge_grad_l2: float
    rejudge_vs_jax_iota: float
    rejudge_vs_jax_g: float
    rejudge_vs_jax_surface_inf: float
    reconstruct_ref_iota: float
    rejudge_vs_reconstruct_surface_inf: float
    runtime: dict[str, object]
    provenance: dict[str, object]

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NestedLsFlatNativeB37Probe:
    """Reduced state at the F3 flat-native B37 (pair2-l2) point."""

    residual_rows: int
    y_star_iota: float
    y_star_g: float
    y_rank: int
    reduced_grad_l2: float
    native_ref_success: bool
    native_ref_iter: int
    native_ref_iota: float
    native_ref_g: float
    native_ref_coil_delta_inf: float
    native_ref_seconds: float
    runtime: dict[str, object]

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NestedLsB37NestedTiming:
    """Process-wall JAX reconstruct walk vs native banana ``run_code``.

    Not an F3 7.70× claim. Requires B3 banana ``run_code`` physics match.
    The two walls are different operators; ``comparable_operators`` is
    always false.
    """

    b3_matched: bool
    comparable_operators: bool
    jax_walk_seconds: float
    jax_walk_iter: int
    jax_walk_iota: float
    jax_walk_success: bool
    native_banana_seconds: float
    native_banana_iter: int
    native_banana_iota: float
    native_banana_success: bool
    coil_delta_inf: float
    runtime: dict[str, object]

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def evaluate_f3_b37_flat_native_probe(
    native: BoozerSurface,
    jax_boozer: BoozerSurfaceJAX,
) -> NestedLsFlatNativeB37Probe:
    """QR ``y*``, reduced gradient, and C++ reconstruct at flat-native B37."""

    residual_fn, _objective_fn, phi_hat = nested_ls_reduced_closures(jax_boozer)
    surface0 = np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64)
    native_coils0 = np.asarray(native.biotsavart.x, dtype=np.float64)
    solution = solve_projected_y(residual_fn, surface0, np.zeros(2, dtype=np.float64))
    require_full_y_rank(solution)
    y_star = np.asarray(solution.solution, dtype=np.float64)
    gradient = np.asarray(reduced_penalty_gradient(phi_hat, surface0), dtype=np.float64)
    native.need_to_run_code = True
    started = time.perf_counter()
    native_ref = native.minimize_boozer_penalty_constraints_newton(
        iota=float(y_star[0]),
        G=float(y_star[1]),
        **nested_ls_physics_newton_kwargs(),
    )
    native_seconds = time.perf_counter() - started
    native_coils1 = np.asarray(native.biotsavart.x, dtype=np.float64)
    return NestedLsFlatNativeB37Probe(
        residual_rows=int(np.asarray(solution.design_matrix).shape[0]),
        y_star_iota=float(y_star[0]),
        y_star_g=float(y_star[1]),
        y_rank=int(np.asarray(solution.numerical_rank)),
        reduced_grad_l2=float(np.linalg.norm(gradient)),
        native_ref_success=bool(native_ref["success"]),
        native_ref_iter=int(native_ref["iter"]),
        native_ref_iota=float(native_ref["iota"]),
        native_ref_g=float(native_ref["G"]),
        native_ref_coil_delta_inf=float(
            np.linalg.norm(native_coils1 - native_coils0, ord=np.inf)
        ),
        native_ref_seconds=float(native_seconds),
        runtime=nested_ls_runtime_identity(),
    )


def evaluate_f3_b37_schur_newton_walk(
    native: BoozerSurface,
    jax_boozer: BoozerSurfaceJAX,
    *,
    maxiter: int = NESTED_LS_NEWTON_MAXITER,
    linear_solver: str = "gmres",
) -> NestedLsSchurNewtonWalkProbe:
    """Frozen-coil Schur Newton walk from F3 B37, then C++ reconstruct rejudge.

    GMRES is the default linear solver; dense_lu is opt-in, not a global
    default. Coils stay frozen.
    """

    residual_fn, _objective_fn, _phi_hat = nested_ls_reduced_closures(jax_boozer)
    surface0 = np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64)
    native_start = np.asarray(native.surface.get_dofs(), dtype=np.float64)
    solution = solve_projected_y(residual_fn, surface0, np.zeros(2, dtype=np.float64))
    require_full_y_rank(solution)
    y_star = np.asarray(solution.solution, dtype=np.float64)
    gradient = np.asarray(
        reduced_penalty_gradient_envelope(_objective_fn, surface0, y_star),
        dtype=np.float64,
    )
    provenance = nested_ls_receipt_provenance()
    runtime = {
        "hostname": provenance["hostname"],
        "python_executable": provenance["python_executable"],
        "jax_default_backend": provenance["jax_default_backend"],
        "jax_devices": provenance["jax_devices"],
        "jax_platforms_env": provenance["jax_platforms_env"],
        "jax_enable_x64_env": provenance["jax_enable_x64_env"],
        "timestamp_utc": provenance["timestamp_utc"],
    }
    newton_kwargs = nested_ls_physics_newton_kwargs()
    native.need_to_run_code = True
    native.surface.set_dofs(native_start)
    reconstruct_ref = native.minimize_boozer_penalty_constraints_newton(
        iota=float(y_star[0]),
        G=float(y_star[1]),
        **newton_kwargs,
    )
    reconstruct_surface = np.asarray(native.surface.get_dofs(), dtype=np.float64)
    reconstruct_ref_iota = float(reconstruct_ref["iota"])
    walk_started = time.perf_counter()
    walk = run_reduced_nested_ls_schur_newton(
        jax_boozer,
        iota=float(y_star[0]),
        G=float(y_star[1]),
        maxiter=int(maxiter),
        linear_solver=str(linear_solver),
    )
    walk_seconds = time.perf_counter() - walk_started
    native.need_to_run_code = True
    native.surface.set_dofs(walk.surface_dofs)
    rejudge_coils0 = np.asarray(native.biotsavart.x, dtype=np.float64)
    rejudge_started = time.perf_counter()
    native_rejudge = native.minimize_boozer_penalty_constraints_newton(
        iota=float(walk.iota),
        G=float(walk.G),
        **newton_kwargs,
    )
    rejudge_seconds = time.perf_counter() - rejudge_started
    rejudge_coils1 = np.asarray(native.biotsavart.x, dtype=np.float64)
    rejudge_surface = np.asarray(native.surface.get_dofs(), dtype=np.float64)
    jax_surface = np.asarray(walk.surface_dofs, dtype=np.float64)
    rejudge_iota = float(native_rejudge["iota"])
    rejudge_g = float(native_rejudge["G"])
    rejudge_grad_l2, _rejudge_inf = _native_ls_gradient_norms(native_rejudge)
    del _rejudge_inf
    return NestedLsSchurNewtonWalkProbe(
        y_star_iota=float(y_star[0]),
        y_star_g=float(y_star[1]),
        reduced_grad_l2_before=float(np.linalg.norm(gradient)),
        maxiter=int(maxiter),
        linear_solver=str(linear_solver),
        iteration_count=int(walk.iteration_count),
        step_accepted=bool(walk.step_accepted),
        success=bool(walk.success),
        persisted=bool(walk.persisted),
        iota=float(walk.iota),
        G=float(walk.G),
        grad_l2=float(np.linalg.norm(walk.reduced_gradient)),
        coil_delta_inf=float(walk.coil_delta_inf),
        walk_seconds=float(walk_seconds),
        gmres_forcing_eta=float(walk.gmres_forcing_eta),
        last_step_eta_achieved=float(walk.steps[-1].gmres_forcing_eta)
        if walk.steps
        else float(walk.gmres_forcing_eta),
        last_step_eta_requested=float(walk.steps[-1].gmres_rtol)
        if walk.steps
        else float(walk.gmres_rtol),
        last_step_forcing_ok=last_step_meets_forcing(walk.steps),
        steps=tuple(asdict(record) for record in walk.steps),
        jax_iota=float(walk.iota),
        jax_g=float(walk.G),
        jax_surface_sha256=sha256_float64(jax_surface),
        jax_surface_dofs=tuple(float(value) for value in jax_surface),
        native_rejudge_success=bool(native_rejudge["success"]),
        native_rejudge_iter=int(native_rejudge["iter"]),
        native_rejudge_iota=rejudge_iota,
        native_rejudge_g=rejudge_g,
        native_rejudge_coil_delta_inf=float(
            np.linalg.norm(rejudge_coils1 - rejudge_coils0, ord=np.inf)
        ),
        native_rejudge_seconds=float(rejudge_seconds),
        native_rejudge_grad_l2=rejudge_grad_l2,
        rejudge_vs_jax_iota=float(rejudge_iota - walk.iota),
        rejudge_vs_jax_g=float(rejudge_g - walk.G),
        rejudge_vs_jax_surface_inf=float(
            np.linalg.norm(rejudge_surface - jax_surface, ord=np.inf)
        ),
        reconstruct_ref_iota=reconstruct_ref_iota,
        rejudge_vs_reconstruct_surface_inf=float(
            np.linalg.norm(rejudge_surface - reconstruct_surface, ord=np.inf)
        ),
        runtime=runtime,
        provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class NestedLsStep2ForcingProbe:
    """Post-step-1 F3 B37 linear-solve η versus Krylov budget and Fourier ``M``.

    Certificate is the independent unpreconditioned
    ``η = ‖(Ĥ+stab I)δ − g‖₂ / ‖g‖₂``. Not a walk, not a timing claim,
    and not F3 7.70×. ``gmres_matvecs`` is unavailable JAX telemetry.
    """

    reduced_grad_l2_before: float
    reduced_grad_l2_after_step1: float
    step1_accepted: bool
    step1_eta_achieved: float
    step1_eta_requested: float
    step2_eta_requested: float
    gmres_restart: int
    coil_delta_inf: float
    factor_seconds: float
    fourier_m_seconds: float | None
    fourier_m_error: str | None
    rows: tuple[dict[str, object], ...]
    any_row_meets_forcing: bool
    runtime: dict[str, object]
    provenance: dict[str, object]

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def _step2_forcing_row(
    matvec,
    rhs: jax.Array,
    *,
    eta_requested: float,
    restart: int,
    maxiter: int,
    maxiter_cap: int,
    preconditioner,
    preconditioner_name: str,
) -> dict[str, object]:
    started = time.perf_counter()
    _delta, _residual, _info, residual_l2, eta, used = (
        solve_operator_gmres_with_forcing(
            matvec,
            rhs,
            eta_requested=float(eta_requested),
            restart=int(restart),
            maxiter=int(maxiter),
            maxiter_cap=int(maxiter_cap),
            preconditioner=preconditioner,
        )
    )
    del _delta, _residual, _info
    return {
        "preconditioner": preconditioner_name,
        "gmres_maxiter": int(maxiter),
        "gmres_maxiter_used": int(used),
        "gmres_maxiter_cap": int(maxiter_cap),
        "gmres_restart": int(restart),
        "eta_achieved": float(eta),
        "eta_requested": float(eta_requested),
        "meets_forcing": bool(linear_solve_meets_forcing(eta, eta_requested)),
        "gmres_residual_l2": float(residual_l2),
        "gmres_seconds": float(time.perf_counter() - started),
    }


def evaluate_f3_b37_step2_forcing_probe(
    jax_boozer: BoozerSurfaceJAX,
) -> NestedLsStep2ForcingProbe:
    """One accepted F3 B37 Schur step, then η vs ``maxiter`` and Fourier ``M``.

    Does not run C++ rejudge and does not walk to 1e-13. Restart stays 8.
    """

    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(jax_boozer)
    del _phi_hat
    surface0 = np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64)
    solution = solve_projected_y(residual_fn, surface0, np.zeros(2, dtype=np.float64))
    require_full_y_rank(solution)
    y_star = np.asarray(solution.solution, dtype=np.float64)
    gradient0 = np.asarray(
        reduced_penalty_gradient_envelope(objective_fn, surface0, y_star),
        dtype=np.float64,
    )
    g0 = float(np.linalg.norm(gradient0))
    step = run_reduced_nested_ls_schur_newton(
        jax_boozer,
        iota=float(y_star[0]),
        G=float(y_star[1]),
        maxiter=1,
    )
    g1 = float(np.linalg.norm(step.reduced_gradient))
    step1 = step.steps[0]
    eta_k = eisenstat_walker_forcing_eta(
        g1,
        previous_grad_norm=g0,
        previous_eta=float(step1.gmres_rtol),
        eta_max=NESTED_LS_SCHUR_GMRES_RTOL,
        nonlinear_tol=NESTED_LS_NEWTON_TOL,
    )
    factor_started = time.perf_counter()
    operator = factor_reduced_nested_ls_schur(
        residual_fn,
        objective_fn,
        step.surface_dofs,
        y_probe=np.array([float(step.iota), float(step.G)], dtype=np.float64),
    )
    factor_seconds = time.perf_counter() - factor_started
    stab = jnp.asarray(NESTED_LS_NEWTON_STAB, dtype=jnp.float64)

    def matvec(tangent: jax.Array) -> jax.Array:
        return operator.apply(tangent) + stab * tangent

    rhs = jnp.asarray(step.reduced_gradient, dtype=jnp.float64)
    restart = int(NESTED_LS_SCHUR_GMRES_RESTART)
    rows = [
        _step2_forcing_row(
            matvec,
            rhs,
            eta_requested=eta_k,
            restart=restart,
            maxiter=budget,
            maxiter_cap=budget,
            preconditioner=None,
            preconditioner_name="none",
        )
        for budget in (8, 16, 32)
    ]
    for cap in (32, NESTED_LS_SCHUR_GMRES_MAXITER_CAP):
        rows.append(
            _step2_forcing_row(
                matvec,
                rhs,
                eta_requested=eta_k,
                restart=restart,
                maxiter=1,
                maxiter_cap=cap,
                preconditioner=None,
                preconditioner_name="none",
            )
        )
    fourier_m_seconds: float | None = None
    fourier_m_error: str | None = None
    try:
        m_started = time.perf_counter()
        names = tuple(jax_boozer.surface.local_full_dof_names)
        preconditioner = factor_schur_fourier_block_preconditioner(
            operator,
            NESTED_LS_NEWTON_STAB,
            names,
            mpol=int(jax_boozer.surface.mpol),
            ntor=int(jax_boozer.surface.ntor),
        )
        fourier_m_seconds = float(time.perf_counter() - m_started)
        for maxiter, cap in ((1, 1), (1, 8)):
            rows.append(
                _step2_forcing_row(
                    matvec,
                    rhs,
                    eta_requested=eta_k,
                    restart=restart,
                    maxiter=maxiter,
                    maxiter_cap=cap,
                    preconditioner=preconditioner.apply,
                    preconditioner_name="fourier_block",
                )
            )
    except NestedLsReducedRankError as error:
        fourier_m_error = str(error)
    provenance = nested_ls_receipt_provenance()
    runtime = {
        "hostname": provenance["hostname"],
        "python_executable": provenance["python_executable"],
        "jax_default_backend": provenance["jax_default_backend"],
        "jax_devices": provenance["jax_devices"],
        "jax_platforms_env": provenance["jax_platforms_env"],
        "jax_enable_x64_env": provenance["jax_enable_x64_env"],
        "timestamp_utc": provenance["timestamp_utc"],
    }
    return NestedLsStep2ForcingProbe(
        reduced_grad_l2_before=g0,
        reduced_grad_l2_after_step1=g1,
        step1_accepted=bool(step.step_accepted),
        step1_eta_achieved=float(step.gmres_forcing_eta),
        step1_eta_requested=float(step1.gmres_rtol),
        step2_eta_requested=float(eta_k),
        gmres_restart=restart,
        coil_delta_inf=float(step.coil_delta_inf),
        factor_seconds=float(factor_seconds),
        fourier_m_seconds=fourier_m_seconds,
        fourier_m_error=fourier_m_error,
        rows=tuple(rows),
        any_row_meets_forcing=any(bool(row["meets_forcing"]) for row in rows),
        runtime=runtime,
        provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class NestedLsStep4ForcingProbe:
    """Frozen step-3 F3 B37 linear-solve η versus GMRES ``maxiter_cap``.

    Persist the surface vector, reload it, and SHA the loaded bytes.
    Historical SHA ``286e3dab…`` is archive only: it is not a replay key.
    Certificate is independent unpreconditioned η. Restart stays 8,
    ``M`` is None. Not a walk, not C++ rejudge, not a timing claim.
    Live requested η and the cap-64 achieved η are distinct.
    """

    surface_sha256: str
    reloaded_surface_sha256: str
    reload_sha_match: bool
    historical_surface_sha256: str
    jax_surface_dofs: tuple[float, ...]
    reduced_grad_l2: float
    iota: float
    G: float
    eta_requested: float
    historical_eta_requested: float
    cap64_eta_achieved: float
    gmres_restart: int
    coil_delta_inf: float
    factor_seconds: float
    rows: tuple[dict[str, object], ...]
    meets_forcing: bool
    fail_closed_reason: str | None
    runtime: dict[str, object]
    provenance: dict[str, object]

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def evaluate_f3_b37_step4_forcing_probe(
    jax_boozer: BoozerSurfaceJAX,
) -> NestedLsStep4ForcingProbe:
    """Persist the JAX step-3 vector, reload it, then raise GMRES ``maxiter``.

    Does not run C++ rejudge. Fourier-block ``M`` is not used.
    """

    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(jax_boozer)
    del _phi_hat
    surface0 = np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64)
    solution = solve_projected_y(residual_fn, surface0, np.zeros(2, dtype=np.float64))
    require_full_y_rank(solution)
    y_star = np.asarray(solution.solution, dtype=np.float64)
    walk = run_reduced_nested_ls_schur_newton(
        jax_boozer,
        iota=float(y_star[0]),
        G=float(y_star[1]),
        maxiter=10,
    )
    frozen = np.array(walk.surface_dofs, dtype=np.float64, copy=True)
    frozen_sha = sha256_float64(frozen)
    jax_boozer.surface.set_dofs(np.zeros_like(frozen))
    jax_boozer.surface.set_dofs(frozen)
    loaded = np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64)
    loaded_sha = sha256_float64(loaded)
    reload_match = loaded_sha == frozen_sha
    g = float(np.linalg.norm(walk.reduced_gradient))
    provenance = nested_ls_receipt_provenance()
    runtime = {
        "hostname": provenance["hostname"],
        "python_executable": provenance["python_executable"],
        "jax_default_backend": provenance["jax_default_backend"],
        "jax_devices": provenance["jax_devices"],
        "jax_platforms_env": provenance["jax_platforms_env"],
        "jax_enable_x64_env": provenance["jax_enable_x64_env"],
        "timestamp_utc": provenance["timestamp_utc"],
    }
    eta_k = float(F3_B37_STEP4_ETA_REQUESTED)
    accepted = [step for step in walk.steps if bool(step.step_accepted)]
    if len(accepted) >= 3:
        eta_k = eisenstat_walker_forcing_eta(
            float(accepted[2].grad_l2),
            previous_grad_norm=float(accepted[1].grad_l2),
            previous_eta=float(accepted[2].gmres_rtol),
            eta_max=NESTED_LS_SCHUR_GMRES_RTOL,
            nonlinear_tol=NESTED_LS_NEWTON_TOL,
        )

    def _empty(reason: str) -> NestedLsStep4ForcingProbe:
        return NestedLsStep4ForcingProbe(
            surface_sha256=frozen_sha,
            reloaded_surface_sha256=loaded_sha,
            reload_sha_match=reload_match,
            historical_surface_sha256=F3_B37_STEP3_SURFACE_SHA256,
            jax_surface_dofs=tuple(float(value) for value in frozen),
            reduced_grad_l2=g,
            iota=float(walk.iota),
            G=float(walk.G),
            eta_requested=float(eta_k),
            historical_eta_requested=float(F3_B37_STEP4_ETA_REQUESTED),
            cap64_eta_achieved=F3_B37_STEP4_CAP64_ETA_ACHIEVED,
            gmres_restart=int(NESTED_LS_SCHUR_GMRES_RESTART),
            coil_delta_inf=float(walk.coil_delta_inf),
            factor_seconds=0.0,
            rows=(),
            meets_forcing=False,
            fail_closed_reason=reason,
            runtime=runtime,
            provenance=provenance,
        )

    if not reload_match:
        return _empty("reload_sha_mismatch")
    if len(accepted) < 3:
        return _empty("step3_not_reproduced")
    if not all(bool(step.step_accepted) for step in accepted[:3]):
        return _empty("step3_not_reproduced")
    if not np.isfinite(g) or g <= 0.0 or not np.isfinite(eta_k):
        return _empty("nonfinite")

    factor_started = time.perf_counter()
    operator = factor_reduced_nested_ls_schur(
        residual_fn,
        objective_fn,
        loaded,
        y_probe=np.array([float(walk.iota), float(walk.G)], dtype=np.float64),
    )
    factor_seconds = time.perf_counter() - factor_started
    stab = jnp.asarray(NESTED_LS_NEWTON_STAB, dtype=jnp.float64)

    def matvec(tangent: jax.Array) -> jax.Array:
        return operator.apply(tangent) + stab * tangent

    rhs = jnp.asarray(walk.reduced_gradient, dtype=jnp.float64)
    restart = int(NESTED_LS_SCHUR_GMRES_RESTART)
    rows: list[dict[str, object]] = []
    previous_residual = float("inf")
    delta: jax.Array | None = None
    fail_reason: str | None = None
    meets = False
    for start_maxiter, cap in ((1, 64), (64, 128), (128, 256), (256, 512)):
        sha_before = sha256_float64(np.asarray(jax_boozer.surface.get_dofs()))
        if sha_before != frozen_sha:
            fail_reason = "state_digest_drift"
            break
        started = time.perf_counter()
        delta, _residual, _info, residual_l2, eta, used = (
            solve_operator_gmres_with_forcing(
                matvec,
                rhs,
                eta_requested=float(eta_k),
                restart=restart,
                maxiter=int(start_maxiter),
                maxiter_cap=int(cap),
                preconditioner=None,
                x0=delta,
            )
        )
        del _residual, _info
        seconds = float(time.perf_counter() - started)
        sha_after = sha256_float64(np.asarray(jax_boozer.surface.get_dofs()))
        finite = bool(np.isfinite(eta) and np.isfinite(residual_l2))
        row = {
            "preconditioner": "none",
            "gmres_maxiter": int(start_maxiter),
            "gmres_maxiter_used": int(used),
            "gmres_maxiter_cap": int(cap),
            "gmres_restart": restart,
            "eta_achieved": float(eta) if finite else None,
            "eta_requested": float(eta_k),
            "historical_eta_requested": float(F3_B37_STEP4_ETA_REQUESTED),
            "cap64_eta_achieved": float(F3_B37_STEP4_CAP64_ETA_ACHIEVED),
            "meets_forcing": bool(finite and linear_solve_meets_forcing(eta, eta_k)),
            "gmres_residual_l2": float(residual_l2) if finite else None,
            "gmres_seconds": seconds,
            "surface_sha256": sha_after,
        }
        rows.append(row)
        if sha_after != frozen_sha:
            fail_reason = "state_digest_drift"
            break
        if not finite:
            fail_reason = "nonfinite"
            break
        if residual_l2 >= previous_residual:
            fail_reason = "stagnation"
            break
        previous_residual = float(residual_l2)
        if linear_solve_meets_forcing(eta, eta_k):
            meets = True
            break
    if not meets and fail_reason is None:
        fail_reason = "eta_unmet"
    return NestedLsStep4ForcingProbe(
        surface_sha256=frozen_sha,
        reloaded_surface_sha256=loaded_sha,
        reload_sha_match=reload_match,
        historical_surface_sha256=F3_B37_STEP3_SURFACE_SHA256,
        jax_surface_dofs=tuple(float(value) for value in frozen),
        reduced_grad_l2=g,
        iota=float(walk.iota),
        G=float(walk.G),
        eta_requested=float(eta_k),
        historical_eta_requested=float(F3_B37_STEP4_ETA_REQUESTED),
        cap64_eta_achieved=F3_B37_STEP4_CAP64_ETA_ACHIEVED,
        gmres_restart=restart,
        coil_delta_inf=float(walk.coil_delta_inf),
        factor_seconds=float(factor_seconds),
        rows=tuple(rows),
        meets_forcing=bool(meets),
        fail_closed_reason=fail_reason,
        runtime=runtime,
        provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class NestedLsStep6ForcingProbe:
    """Frozen step-5 F3 B37 linear-solve η versus GMRES cap 512 then 1024.

    Load the cap-512 walk vector, clobber, reload, and SHA the loaded
    bytes. Certificate is independent unpreconditioned η. Restart stays
    8, ``M`` is None. 2048 starts at the cap (not at 1024) and runs
    only if residual falls by the predeclared ratio and the start-at-cap
    predictor stays under the wall. Not a walk, not C++ rejudge, not a
    timing claim.
    """

    surface_sha256: str
    reloaded_surface_sha256: str
    reload_sha_match: bool
    reduced_grad_l2: float
    iota: float
    G: float
    eta_requested: float
    cap512_eta_achieved: float
    gmres_restart: int
    coil_delta_inf: float
    factor_seconds: float
    wall_seconds_2048: float
    material_residual_ratio: float
    rows: tuple[dict[str, object], ...]
    meets_forcing: bool
    fail_closed_reason: str | None
    unpreconditioned_gmres_insufficient: bool
    runtime: dict[str, object]
    provenance: dict[str, object]

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def evaluate_f3_b37_step6_forcing_probe(
    jax_boozer: BoozerSurfaceJAX,
) -> NestedLsStep6ForcingProbe:
    """Load the persisted step-5 vector and raise GMRES ``maxiter``.

    Does not run C++ rejudge or a Newton walk. Fourier-block ``M`` is
    not used.
    """

    if not F3_B37_CAP512_WALK_EVIDENCE.is_file():
        raise FileNotFoundError(str(F3_B37_CAP512_WALK_EVIDENCE))
    archived = json.loads(F3_B37_CAP512_WALK_EVIDENCE.read_text(encoding="utf-8"))
    archived_probe = archived["probe"]
    frozen = np.ascontiguousarray(
        np.asarray(archived_probe["jax_surface_dofs"], dtype=np.float64)
    )
    stored_sha = str(archived_probe["jax_surface_sha256"])
    frozen_sha = sha256_float64(frozen)
    provenance = nested_ls_receipt_provenance()
    runtime = {
        "hostname": provenance["hostname"],
        "python_executable": provenance["python_executable"],
        "jax_default_backend": provenance["jax_default_backend"],
        "jax_devices": provenance["jax_devices"],
        "jax_platforms_env": provenance["jax_platforms_env"],
        "jax_enable_x64_env": provenance["jax_enable_x64_env"],
        "timestamp_utc": provenance["timestamp_utc"],
    }

    def _empty(
        reason: str,
        *,
        loaded_sha: str,
        reload_match: bool,
        g_value: float,
        iota: float,
        g_const: float,
        coil: float,
    ) -> NestedLsStep6ForcingProbe:
        return NestedLsStep6ForcingProbe(
            surface_sha256=frozen_sha,
            reloaded_surface_sha256=loaded_sha,
            reload_sha_match=reload_match,
            reduced_grad_l2=g_value,
            iota=float(iota),
            G=float(g_const),
            eta_requested=float(F3_B37_STEP6_ETA_REQUESTED),
            cap512_eta_achieved=float(F3_B37_STEP6_CAP512_ETA_ACHIEVED),
            gmres_restart=int(NESTED_LS_SCHUR_GMRES_RESTART),
            coil_delta_inf=float(coil),
            factor_seconds=0.0,
            wall_seconds_2048=float(F3_B37_STEP6_WALL_SECONDS_2048),
            material_residual_ratio=float(F3_B37_STEP6_MATERIAL_RESIDUAL_RATIO),
            rows=(),
            meets_forcing=False,
            fail_closed_reason=reason,
            unpreconditioned_gmres_insufficient=unpreconditioned_gmres_is_insufficient(
                reason
            ),
            runtime=runtime,
            provenance=provenance,
        )

    if frozen.size != 661 or frozen_sha != F3_B37_STEP5_SURFACE_SHA256:
        return _empty(
            "surface_sha_mismatch",
            loaded_sha=frozen_sha,
            reload_match=False,
            g_value=0.0,
            iota=0.0,
            g_const=0.0,
            coil=0.0,
        )
    if stored_sha != F3_B37_STEP5_SURFACE_SHA256:
        return _empty(
            "surface_sha_mismatch",
            loaded_sha=stored_sha,
            reload_match=False,
            g_value=0.0,
            iota=0.0,
            g_const=0.0,
            coil=0.0,
        )
    jax_boozer.surface.set_dofs(np.zeros_like(frozen))
    jax_boozer.surface.set_dofs(frozen)
    loaded = np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64)
    loaded_sha = sha256_float64(loaded)
    reload_match = loaded_sha == frozen_sha
    coils_before = np.asarray(jax_boozer.biotsavart.x, dtype=np.float64)
    if not reload_match:
        return _empty(
            "reload_sha_mismatch",
            loaded_sha=loaded_sha,
            reload_match=False,
            g_value=0.0,
            iota=F3_B37_STEP6_IOTA,
            g_const=F3_B37_STEP6_G,
            coil=0.0,
        )
    iota = float(F3_B37_STEP6_IOTA)
    g_const = float(F3_B37_STEP6_G)
    if abs(float(archived_probe["iota"]) - iota) > 1.0e-15:
        return _empty(
            "iota_mismatch",
            loaded_sha=loaded_sha,
            reload_match=True,
            g_value=0.0,
            iota=float(archived_probe["iota"]),
            g_const=g_const,
            coil=0.0,
        )
    if abs(float(archived_probe["G"]) - g_const) > 1.0e-15:
        return _empty(
            "g_mismatch",
            loaded_sha=loaded_sha,
            reload_match=True,
            g_value=0.0,
            iota=iota,
            g_const=float(archived_probe["G"]),
            coil=0.0,
        )
    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(jax_boozer)
    del _phi_hat
    y = np.array([iota, g_const], dtype=np.float64)
    y_star = solve_projected_y(residual_fn, loaded, y).solution
    gradient = np.asarray(
        reduced_penalty_gradient_envelope(objective_fn, loaded, y_star),
        dtype=np.float64,
    )
    g_value = float(np.linalg.norm(gradient))
    coil = float(
        np.linalg.norm(
            np.asarray(jax_boozer.biotsavart.x, dtype=np.float64) - coils_before,
            ord=np.inf,
        )
    )
    if not np.isfinite(g_value) or abs(g_value - F3_B37_STEP6_GRAD_L2) > 1.0e-12:
        return _empty(
            "grad_mismatch",
            loaded_sha=loaded_sha,
            reload_match=True,
            g_value=g_value,
            iota=iota,
            g_const=g_const,
            coil=coil,
        )
    eta_k = float(F3_B37_STEP6_ETA_REQUESTED)
    factor_started = time.perf_counter()
    operator = factor_reduced_nested_ls_schur(
        residual_fn,
        objective_fn,
        loaded,
        y_probe=np.asarray(y_star, dtype=np.float64),
    )
    factor_seconds = time.perf_counter() - factor_started
    stab = jnp.asarray(NESTED_LS_NEWTON_STAB, dtype=jnp.float64)

    def matvec(tangent: jax.Array) -> jax.Array:
        return operator.apply(tangent) + stab * tangent

    rhs = jnp.asarray(gradient, dtype=jnp.float64)
    restart = int(NESTED_LS_SCHUR_GMRES_RESTART)
    rows: list[dict[str, object]] = []
    previous_residual = float("inf")
    previous_seconds = 0.0
    delta: jax.Array | None = None
    fail_reason: str | None = None
    meets = False
    schedule = ((1, 512), (512, 1024))
    for start_maxiter, cap in schedule:
        sha_before = sha256_float64(np.asarray(jax_boozer.surface.get_dofs()))
        if sha_before != frozen_sha:
            fail_reason = "state_digest_drift"
            break
        started = time.perf_counter()
        delta, _residual, _info, residual_l2, eta, used = (
            solve_operator_gmres_with_forcing(
                matvec,
                rhs,
                eta_requested=eta_k,
                restart=restart,
                maxiter=int(start_maxiter),
                maxiter_cap=int(cap),
                preconditioner=None,
                x0=delta,
            )
        )
        del _residual, _info
        seconds = float(time.perf_counter() - started)
        sha_after = sha256_float64(np.asarray(jax_boozer.surface.get_dofs()))
        finite = bool(np.isfinite(eta) and np.isfinite(residual_l2))
        row = {
            "preconditioner": "none",
            "gmres_maxiter": int(start_maxiter),
            "gmres_maxiter_used": int(used),
            "gmres_maxiter_cap": int(cap),
            "gmres_restart": restart,
            "eta_achieved": float(eta) if finite else None,
            "eta_requested": eta_k,
            "cap512_eta_achieved": float(F3_B37_STEP6_CAP512_ETA_ACHIEVED),
            "meets_forcing": bool(finite and linear_solve_meets_forcing(eta, eta_k)),
            "gmres_residual_l2": float(residual_l2) if finite else None,
            "gmres_seconds": seconds,
            "surface_sha256": sha_after,
        }
        rows.append(row)
        if sha_after != frozen_sha:
            fail_reason = "state_digest_drift"
            break
        if not finite:
            fail_reason = "nonfinite"
            break
        if residual_l2 >= previous_residual:
            fail_reason = "stagnation"
            break
        previous_residual = float(residual_l2)
        previous_seconds = seconds
        if linear_solve_meets_forcing(eta, eta_k):
            meets = True
            break
    last_start = _as_int(rows[-1]["gmres_maxiter"]) if rows else 0
    last_cap = _as_int(rows[-1]["gmres_maxiter_cap"]) if rows else 0
    predicted_2048 = predict_start_at_cap_wall_seconds(
        previous_seconds,
        previous_start_maxiter=last_start,
        previous_cap=last_cap,
        next_cap=F3_B37_STEP6_CAP_2048,
    )
    residual_fell = bool(
        rows
        and previous_residual
        < F3_B37_STEP6_MATERIAL_RESIDUAL_RATIO * _as_float(rows[0]["gmres_residual_l2"])
    )
    if (
        not meets
        and fail_reason is None
        and residual_fell
        and predicted_2048 <= F3_B37_STEP6_WALL_SECONDS_2048
    ):
        sha_before = sha256_float64(np.asarray(jax_boozer.surface.get_dofs()))
        if sha_before != frozen_sha:
            fail_reason = "state_digest_drift"
        else:
            started = time.perf_counter()
            _delta, _residual, _info, residual_l2, eta, used = (
                solve_operator_gmres_with_forcing(
                    matvec,
                    rhs,
                    eta_requested=eta_k,
                    restart=restart,
                    maxiter=int(F3_B37_STEP6_CAP_2048_START_MAXITER),
                    maxiter_cap=int(F3_B37_STEP6_CAP_2048),
                    preconditioner=None,
                    x0=delta,
                )
            )
            del _delta, _residual, _info
            seconds = float(time.perf_counter() - started)
            sha_after = sha256_float64(np.asarray(jax_boozer.surface.get_dofs()))
            finite = bool(np.isfinite(eta) and np.isfinite(residual_l2))
            rows.append(
                {
                    "preconditioner": "none",
                    "gmres_maxiter": int(F3_B37_STEP6_CAP_2048_START_MAXITER),
                    "gmres_maxiter_used": int(used),
                    "gmres_maxiter_cap": int(F3_B37_STEP6_CAP_2048),
                    "gmres_restart": restart,
                    "eta_achieved": float(eta) if finite else None,
                    "eta_requested": eta_k,
                    "cap512_eta_achieved": float(F3_B37_STEP6_CAP512_ETA_ACHIEVED),
                    "meets_forcing": bool(
                        finite and linear_solve_meets_forcing(eta, eta_k)
                    ),
                    "gmres_residual_l2": float(residual_l2) if finite else None,
                    "gmres_seconds": seconds,
                    "surface_sha256": sha_after,
                    "wall_seconds_cap": float(F3_B37_STEP6_WALL_SECONDS_2048),
                    "predicted_seconds": float(predicted_2048),
                }
            )
            if sha_after != frozen_sha:
                fail_reason = "state_digest_drift"
            elif not finite:
                fail_reason = "nonfinite"
            elif residual_l2 >= previous_residual:
                fail_reason = "stagnation"
            elif linear_solve_meets_forcing(eta, eta_k):
                meets = True
            elif seconds > F3_B37_STEP6_WALL_SECONDS_2048:
                fail_reason = "wall_time_cap"
    elif not meets and fail_reason is None and rows:
        resid_512 = _as_float(rows[0]["gmres_residual_l2"])
        if previous_residual >= F3_B37_STEP6_MATERIAL_RESIDUAL_RATIO * resid_512:
            fail_reason = "stagnation"
        elif predicted_2048 > F3_B37_STEP6_WALL_SECONDS_2048:
            fail_reason = "wall_time_cap"
    if not meets and fail_reason is None:
        fail_reason = "eta_unmet"
    first_residual = (
        _as_float(rows[0]["gmres_residual_l2"])
        if rows and rows[0].get("gmres_residual_l2") is not None
        else None
    )
    residual_ratio = (
        float(previous_residual) / first_residual
        if first_residual is not None
        and first_residual > 0.0
        and np.isfinite(previous_residual)
        else None
    )
    return NestedLsStep6ForcingProbe(
        surface_sha256=frozen_sha,
        reloaded_surface_sha256=loaded_sha,
        reload_sha_match=reload_match,
        reduced_grad_l2=g_value,
        iota=iota,
        G=g_const,
        eta_requested=eta_k,
        cap512_eta_achieved=float(F3_B37_STEP6_CAP512_ETA_ACHIEVED),
        gmres_restart=restart,
        coil_delta_inf=coil,
        factor_seconds=float(factor_seconds),
        wall_seconds_2048=float(F3_B37_STEP6_WALL_SECONDS_2048),
        material_residual_ratio=float(F3_B37_STEP6_MATERIAL_RESIDUAL_RATIO),
        rows=tuple(rows),
        meets_forcing=bool(meets),
        fail_closed_reason=fail_reason,
        unpreconditioned_gmres_insufficient=unpreconditioned_gmres_is_insufficient(
            fail_reason,
            residual_ratio=residual_ratio,
            material_residual_ratio=float(F3_B37_STEP6_MATERIAL_RESIDUAL_RATIO),
        ),
        runtime=runtime,
        provenance=provenance,
    )


def _json_finite(value: object) -> float | None:
    """JSON-safe float; ``None`` when the value is non-finite."""

    number = _as_float(value)
    if not np.isfinite(number):
        return None
    return number


def _live_unpreconditioned_eta(
    matvec: Callable[[jax.Array], jax.Array],
    delta: jax.Array,
    rhs: jax.Array,
) -> tuple[float, float]:
    """Independent ``‖Aδ − b‖₂ / ‖b‖₂`` against the live operator."""

    residual = jax.block_until_ready(matvec(delta) - rhs)
    residual_l2 = float(
        np.linalg.norm(np.asarray(jax.device_get(residual), dtype=np.float64))
    )
    rhs_l2 = float(np.linalg.norm(np.asarray(jax.device_get(rhs), dtype=np.float64)))
    eta = residual_l2 / rhs_l2 if rhs_l2 > 0.0 else 0.0
    return residual_l2, eta


def _dense_schur_spectrum(
    dense_np: np.ndarray,
    rhs_np: np.ndarray,
    delta_np: np.ndarray,
) -> dict[str, object]:
    """Symmetry defect, eig extrema, and Rayleigh quotients of Ĥ_ss+stab I."""

    frobenius = float(np.linalg.norm(dense_np, ord="fro"))
    defect = float(np.linalg.norm(dense_np - dense_np.T, ord="fro"))
    eigenvalues = np.linalg.eigvals(dense_np)
    real = np.real(eigenvalues)
    imag = np.imag(eigenvalues)
    abs_eigs = np.abs(eigenvalues)
    abs_min = float(np.min(abs_eigs))
    abs_max = float(np.max(abs_eigs))
    condition = abs_max / abs_min if abs_min > 0.0 else None

    def _rayleigh(vector: np.ndarray) -> float | None:
        scale = float(np.linalg.norm(vector))
        if scale <= 0.0 or not np.isfinite(scale):
            return None
        unit = vector / scale
        return _json_finite(float(unit @ (dense_np @ unit)))

    return {
        "dimension": int(dense_np.shape[0]),
        "symmetry_frobenius": _json_finite(defect),
        "symmetry_frobenius_rel": _json_finite(
            defect / frobenius if frobenius > 0.0 else float("nan")
        ),
        "eig_real_min": _json_finite(float(np.min(real))),
        "eig_real_max": _json_finite(float(np.max(real))),
        "eig_abs_min": _json_finite(abs_min),
        "eig_abs_max": _json_finite(abs_max),
        "eig_condition": _json_finite(condition) if condition is not None else None,
        "n_eig_negative_real": int(np.sum(real < 0.0)),
        "n_eig_complex": int(np.sum(np.abs(imag) > 1.0e-12)),
        "rayleigh_rhs": _rayleigh(rhs_np),
        "rayleigh_delta": _rayleigh(delta_np),
    }


@dataclass(frozen=True, slots=True)
class NestedLsStep6ArchitectureProbe:
    """Frozen step-6 solver-architecture canary.

    Dense LU, dense-inverse ``M`` (option B), counted restart/method
    sweep, and a spectral snapshot of ``Ĥ_ss+stab I``. Not a walk, not
    cap-2048, not a timing claim, and not F3 7.70×.
    """

    surface_sha256: str
    reloaded_surface_sha256: str
    reload_sha_match: bool
    reduced_grad_l2: float
    iota: float
    G: float
    eta_requested: float
    coil_delta_inf: float
    factor_seconds: float
    dense_bytes: int
    dense_chunk_batch_size: int
    dense_chunk_batch_size_env: str | None
    wall_seconds: float
    hvp_budget: int
    spectrum: dict[str, object]
    rows: tuple[dict[str, object], ...]
    dense_meets_forcing: bool
    fail_closed_reason: str | None
    runtime: dict[str, object]
    provenance: dict[str, object]

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def evaluate_f3_b37_step6_architecture_probe(
    jax_boozer: BoozerSurfaceJAX,
) -> NestedLsStep6ArchitectureProbe:
    """Dense-LU canary and counted GMRES sweep at the frozen step-6 vector.

    Does not run C++ rejudge, a Newton walk, or cap-2048. Fourier-block
    ``M`` is not used. Certificates are independent live-matvec
    unpreconditioned η.
    """

    if not F3_B37_CAP512_WALK_EVIDENCE.is_file():
        raise FileNotFoundError(str(F3_B37_CAP512_WALK_EVIDENCE))
    archived = json.loads(F3_B37_CAP512_WALK_EVIDENCE.read_text(encoding="utf-8"))
    archived_probe = archived["probe"]
    frozen = np.ascontiguousarray(
        np.asarray(archived_probe["jax_surface_dofs"], dtype=np.float64)
    )
    stored_sha = str(archived_probe["jax_surface_sha256"])
    frozen_sha = sha256_float64(frozen)
    provenance = nested_ls_receipt_provenance()
    runtime = {
        "hostname": provenance["hostname"],
        "python_executable": provenance["python_executable"],
        "jax_default_backend": provenance["jax_default_backend"],
        "jax_devices": provenance["jax_devices"],
        "jax_platforms_env": provenance["jax_platforms_env"],
        "jax_enable_x64_env": provenance["jax_enable_x64_env"],
        "timestamp_utc": provenance["timestamp_utc"],
    }
    eta_k = float(F3_B37_STEP6_ETA_REQUESTED)
    dense_chunk_batch_size = dense_operator_chunk_batch_size()
    dense_chunk_batch_size_env = os.environ.get(
        "SIMSOPT_DENSE_OPERATOR_CHUNK_BATCH_SIZE"
    )
    empty_spectrum: dict[str, object] = {
        "dimension": 0,
        "symmetry_frobenius": None,
        "symmetry_frobenius_rel": None,
        "eig_real_min": None,
        "eig_real_max": None,
        "eig_abs_min": None,
        "eig_abs_max": None,
        "eig_condition": None,
        "n_eig_negative_real": 0,
        "n_eig_complex": 0,
        "rayleigh_rhs": None,
        "rayleigh_delta": None,
    }

    def _empty(
        reason: str,
        *,
        loaded_sha: str,
        reload_match: bool,
        g_value: float,
        iota: float,
        g_const: float,
        coil: float,
    ) -> NestedLsStep6ArchitectureProbe:
        grad_json = _json_finite(g_value)
        iota_json = _json_finite(iota)
        g_json = _json_finite(g_const)
        coil_json = _json_finite(coil)
        return NestedLsStep6ArchitectureProbe(
            surface_sha256=frozen_sha,
            reloaded_surface_sha256=loaded_sha,
            reload_sha_match=reload_match,
            reduced_grad_l2=0.0 if grad_json is None else grad_json,
            iota=0.0 if iota_json is None else iota_json,
            G=0.0 if g_json is None else g_json,
            eta_requested=eta_k,
            coil_delta_inf=0.0 if coil_json is None else coil_json,
            factor_seconds=0.0,
            dense_bytes=schur_dense_operator_bytes(int(frozen.size)),
            dense_chunk_batch_size=dense_chunk_batch_size,
            dense_chunk_batch_size_env=dense_chunk_batch_size_env,
            wall_seconds=float(F3_B37_STEP6_ARCH_WALL_SECONDS),
            hvp_budget=int(F3_B37_STEP6_ARCH_HVP_BUDGET),
            spectrum=dict(empty_spectrum),
            rows=(),
            dense_meets_forcing=False,
            fail_closed_reason=reason,
            runtime=runtime,
            provenance=provenance,
        )

    if frozen.size != 661 or frozen_sha != F3_B37_STEP5_SURFACE_SHA256:
        return _empty(
            "surface_sha_mismatch",
            loaded_sha=frozen_sha,
            reload_match=False,
            g_value=0.0,
            iota=0.0,
            g_const=0.0,
            coil=0.0,
        )
    if stored_sha != F3_B37_STEP5_SURFACE_SHA256:
        return _empty(
            "surface_sha_mismatch",
            loaded_sha=stored_sha,
            reload_match=False,
            g_value=0.0,
            iota=0.0,
            g_const=0.0,
            coil=0.0,
        )
    jax_boozer.surface.set_dofs(np.zeros_like(frozen))
    jax_boozer.surface.set_dofs(frozen)
    loaded = np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64)
    loaded_sha = sha256_float64(loaded)
    reload_match = loaded_sha == frozen_sha
    coils_before = np.asarray(jax_boozer.biotsavart.x, dtype=np.float64)
    if not reload_match:
        return _empty(
            "reload_sha_mismatch",
            loaded_sha=loaded_sha,
            reload_match=False,
            g_value=0.0,
            iota=F3_B37_STEP6_IOTA,
            g_const=F3_B37_STEP6_G,
            coil=0.0,
        )
    iota = float(F3_B37_STEP6_IOTA)
    g_const = float(F3_B37_STEP6_G)
    if abs(float(archived_probe["iota"]) - iota) > 1.0e-15:
        return _empty(
            "iota_mismatch",
            loaded_sha=loaded_sha,
            reload_match=True,
            g_value=0.0,
            iota=float(archived_probe["iota"]),
            g_const=g_const,
            coil=0.0,
        )
    if abs(float(archived_probe["G"]) - g_const) > 1.0e-15:
        return _empty(
            "g_mismatch",
            loaded_sha=loaded_sha,
            reload_match=True,
            g_value=0.0,
            iota=iota,
            g_const=float(archived_probe["G"]),
            coil=0.0,
        )
    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(jax_boozer)
    del _phi_hat
    y = np.array([iota, g_const], dtype=np.float64)
    y_star = solve_projected_y(residual_fn, loaded, y).solution
    gradient = np.asarray(
        reduced_penalty_gradient_envelope(objective_fn, loaded, y_star),
        dtype=np.float64,
    )
    g_value = float(np.linalg.norm(gradient))
    coil = float(
        np.linalg.norm(
            np.asarray(jax_boozer.biotsavart.x, dtype=np.float64) - coils_before,
            ord=np.inf,
        )
    )
    if not np.isfinite(g_value) or abs(g_value - F3_B37_STEP6_GRAD_L2) > 1.0e-12:
        return _empty(
            "grad_mismatch",
            loaded_sha=loaded_sha,
            reload_match=True,
            g_value=g_value,
            iota=iota,
            g_const=g_const,
            coil=coil,
        )
    factor_started = time.perf_counter()
    operator = factor_reduced_nested_ls_schur(
        residual_fn,
        objective_fn,
        loaded,
        y_probe=np.asarray(y_star, dtype=np.float64),
    )
    factor_seconds = time.perf_counter() - factor_started
    stab = jnp.asarray(NESTED_LS_NEWTON_STAB, dtype=jnp.float64)

    def live_matvec(tangent: jax.Array) -> jax.Array:
        return operator.apply(tangent) + stab * tangent

    rhs = jnp.asarray(gradient, dtype=jnp.float64)
    dense_bytes = schur_dense_operator_bytes(int(frozen.size))
    deadline = time.perf_counter() + float(F3_B37_STEP6_ARCH_WALL_SECONDS)
    rows: list[dict[str, object]] = []
    spectrum = dict(empty_spectrum)
    dense_meets = False
    fail_reason: str | None = None
    dense_np: np.ndarray | None = None
    dense_jax: jax.Array | None = None

    def _surface_sha() -> str:
        return sha256_float64(np.asarray(jax_boozer.surface.get_dofs()))

    def _counted_gmres_row(
        *,
        krylov_matvec: Callable[[jax.Array], jax.Array],
        restart: int,
        maxiter: int,
        solve_method: str,
        preconditioner: Callable[[jax.Array], jax.Array] | None,
        jax_tol: float,
        operator_name: str,
        row_name: str,
    ) -> dict[str, object]:
        sha_before = _surface_sha()
        counter = NestedLsCountedMatvec(krylov_matvec)
        started = time.perf_counter()
        with jax.transfer_guard_host_to_device("allow"):
            delta, info = jax_gmres(
                counter,
                rhs,
                None,
                tol=float(jax_tol),
                atol=0.0,
                restart=int(restart),
                maxiter=int(maxiter),
                M=preconditioner,
                solve_method=str(solve_method),
            )
        delta = jax.block_until_ready(delta)
        seconds = float(time.perf_counter() - started)
        residual_l2, eta = _live_unpreconditioned_eta(live_matvec, delta, rhs)
        info_host = int(
            np.asarray(
                jax.device_get(jnp.reshape(jnp.asarray(info), (1,))),
                dtype=np.int32,
            )[0]
        )
        finite = bool(np.isfinite(eta) and np.isfinite(residual_l2))
        sha_after = _surface_sha()
        return {
            "row": row_name,
            "linear_solver": "gmres",
            "operator": operator_name,
            "preconditioner": "none" if preconditioner is None else "dense_inverse",
            "solve_method": str(solve_method),
            "gmres_restart": int(restart),
            "gmres_maxiter": int(maxiter),
            "gmres_info": info_host,
            "operator_applications": int(counter.count),
            "eta_achieved": _json_finite(eta),
            "eta_requested": eta_k,
            "meets_forcing": bool(finite and linear_solve_meets_forcing(eta, eta_k)),
            "gmres_residual_l2": _json_finite(residual_l2),
            "gmres_seconds": seconds,
            "surface_sha256": sha_after,
            "state_digest_drift": bool(
                sha_before != frozen_sha or sha_after != frozen_sha
            ),
        }

    if _surface_sha() != frozen_sha:
        fail_reason = "state_digest_drift"
    else:
        dense_started = time.perf_counter()
        materialized: jax.Array = jax.block_until_ready(
            materialize_stabilized_schur_dense(operator, float(NESTED_LS_NEWTON_STAB))
        )
        dense_jax = materialized
        dense_seconds = float(time.perf_counter() - dense_started)
        lu_started = time.perf_counter()
        delta_lu = jax.block_until_ready(
            solve_stabilized_schur_dense_lu(materialized, rhs)
        )
        lu_seconds = float(time.perf_counter() - lu_started)
        live_residual_l2, live_eta = _live_unpreconditioned_eta(
            live_matvec, delta_lu, rhs
        )
        dense_residual = jax.block_until_ready(materialized @ delta_lu - rhs)
        dense_residual_l2 = float(
            np.linalg.norm(np.asarray(jax.device_get(dense_residual), dtype=np.float64))
        )
        dense_eta = dense_residual_l2 / g_value if g_value > 0.0 else 0.0
        finite_lu = bool(np.isfinite(live_eta) and np.isfinite(live_residual_l2))
        dense_meets = bool(finite_lu and linear_solve_meets_forcing(live_eta, eta_k))
        sha_after_dense = _surface_sha()
        rows.append(
            {
                "row": "dense_lu",
                "linear_solver": "dense_lu",
                "operator": "chunked_dense",
                "preconditioner": "none",
                "solve_method": None,
                "gmres_restart": None,
                "gmres_maxiter": None,
                "gmres_info": 0,
                "operator_applications": int(frozen.size),
                "eta_achieved": _json_finite(live_eta),
                "eta_achieved_dense_materialization": _json_finite(dense_eta),
                "eta_requested": eta_k,
                "meets_forcing": dense_meets,
                "gmres_residual_l2": _json_finite(live_residual_l2),
                "dense_residual_l2": _json_finite(dense_residual_l2),
                "dense_seconds": dense_seconds,
                "lu_seconds": lu_seconds,
                "gmres_seconds": dense_seconds + lu_seconds,
                "dense_bytes": dense_bytes,
                "dense_chunk_batch_size": dense_chunk_batch_size,
                "dense_chunk_batch_size_env": dense_chunk_batch_size_env,
                "surface_sha256": sha_after_dense,
                "state_digest_drift": bool(sha_after_dense != frozen_sha),
            }
        )
        print(
            "arch row dense_lu"
            f" eta_live={live_eta!r} eta_dense={dense_eta!r}"
            f" dense_s={dense_seconds:.3f} lu_s={lu_seconds:.3f}",
            flush=True,
        )
        if sha_after_dense != frozen_sha:
            fail_reason = "state_digest_drift"
        elif not finite_lu:
            fail_reason = "nonfinite"
        else:
            dense_np = np.asarray(jax.device_get(materialized), dtype=np.float64)
            delta_np = np.asarray(jax.device_get(delta_lu), dtype=np.float64)
            spectrum = _dense_schur_spectrum(dense_np, gradient, delta_np)

    def dense_matvec(tangent: jax.Array) -> jax.Array:
        if dense_jax is None:
            raise RuntimeError("dense_matvec requires a materialized Schur matrix")
        return dense_jax @ tangent

    if fail_reason is None and dense_jax is not None and time.perf_counter() < deadline:
        apply_m = dense_schur_inverse_preconditioner(dense_jax)
        row = _counted_gmres_row(
            krylov_matvec=live_matvec,
            restart=int(NESTED_LS_SCHUR_GMRES_RESTART),
            maxiter=1,
            solve_method="incremental",
            preconditioner=apply_m,
            jax_tol=eta_k,
            operator_name="live_schur_hvp",
            row_name="option_b_dense_inverse_m",
        )
        row["shared_dense_assembly"] = True
        row["excludes_assembly_seconds"] = True
        row["excludes_inversion_seconds"] = True
        rows.append(row)
        print(
            f"arch row {row['row']}"
            f" eta={row['eta_achieved']!r} apps={row['operator_applications']}"
            f" s={row['gmres_seconds']:.3f}",
            flush=True,
        )
        if bool(row["state_digest_drift"]):
            fail_reason = "state_digest_drift"
        elif row["eta_achieved"] is None:
            fail_reason = "nonfinite"

    if fail_reason is None and dense_jax is not None and time.perf_counter() < deadline:
        row = _counted_gmres_row(
            krylov_matvec=dense_matvec,
            restart=int(F3_B37_STEP6_ARCH_FULL_RESTART),
            maxiter=1,
            solve_method="incremental",
            preconditioner=None,
            jax_tol=float(F3_B37_STEP6_ARCH_SWEEP_JAX_TOL),
            operator_name="dense_matvec",
            row_name="full_gmres_dense_matvec",
        )
        row["shared_dense_assembly"] = True
        row["excludes_assembly_seconds"] = True
        rows.append(row)
        print(
            f"arch row {row['row']}"
            f" eta={row['eta_achieved']!r} apps={row['operator_applications']}"
            f" s={row['gmres_seconds']:.3f}",
            flush=True,
        )
        if bool(row["state_digest_drift"]):
            fail_reason = "state_digest_drift"
        elif row["eta_achieved"] is None:
            fail_reason = "nonfinite"

    for restart in F3_B37_STEP6_ARCH_RESTARTS:
        maxiter = max(1, int(F3_B37_STEP6_ARCH_HVP_BUDGET) // int(restart))
        for solve_method in F3_B37_STEP6_ARCH_SOLVE_METHODS:
            if fail_reason is not None or time.perf_counter() >= deadline:
                if fail_reason is None:
                    fail_reason = "wall_time_cap"
                break
            row = _counted_gmres_row(
                krylov_matvec=live_matvec,
                restart=int(restart),
                maxiter=int(maxiter),
                solve_method=str(solve_method),
                preconditioner=None,
                jax_tol=float(F3_B37_STEP6_ARCH_SWEEP_JAX_TOL),
                operator_name="live_schur_hvp",
                row_name=f"live_restart_{restart}_{solve_method}",
            )
            rows.append(row)
            print(
                f"arch row {row['row']}"
                f" eta={row['eta_achieved']!r} apps={row['operator_applications']}"
                f" s={row['gmres_seconds']:.3f}",
                flush=True,
            )
            if bool(row["state_digest_drift"]):
                fail_reason = "state_digest_drift"
                break
            if row["eta_achieved"] is None:
                fail_reason = "nonfinite"
                break
        if fail_reason is not None:
            break

    if fail_reason is None and time.perf_counter() >= deadline:
        fail_reason = "wall_time_cap"

    return NestedLsStep6ArchitectureProbe(
        surface_sha256=frozen_sha,
        reloaded_surface_sha256=loaded_sha,
        reload_sha_match=reload_match,
        reduced_grad_l2=g_value,
        iota=iota,
        G=g_const,
        eta_requested=eta_k,
        coil_delta_inf=coil,
        factor_seconds=float(factor_seconds),
        dense_bytes=int(dense_bytes),
        dense_chunk_batch_size=dense_chunk_batch_size,
        dense_chunk_batch_size_env=dense_chunk_batch_size_env,
        wall_seconds=float(F3_B37_STEP6_ARCH_WALL_SECONDS),
        hvp_budget=int(F3_B37_STEP6_ARCH_HVP_BUDGET),
        spectrum=spectrum,
        rows=tuple(rows),
        dense_meets_forcing=bool(dense_meets),
        fail_closed_reason=fail_reason,
        runtime=runtime,
        provenance=provenance,
    )


def _empty_dense_spectrum() -> dict[str, object]:
    return {
        "dimension": 0,
        "symmetry_frobenius": None,
        "symmetry_frobenius_rel": None,
        "eig_real_min": None,
        "eig_real_max": None,
        "eig_abs_min": None,
        "eig_abs_max": None,
        "eig_condition": None,
        "n_eig_negative_real": 0,
        "n_eig_complex": 0,
        "rayleigh_rhs": None,
        "rayleigh_delta": None,
    }


@dataclass(frozen=True, slots=True)
class NestedLsEndpointAdjointProbe:
    """Unregularized IFT adjoint canary at the dense-LU walk endpoint.

    Opt-in past the 1 MiB stored-matrix cap. Uses ``stab=0`` Ĥ_ss, not
    the Newton ``stab=1e-4`` factor. Not a walk, not cap-2048, not a
    default switch, not B3, and not F3 7.70×.
    """

    surface_sha256: str
    reloaded_surface_sha256: str
    reload_sha_match: bool
    reduced_grad_l2: float
    iota: float
    G: float
    coil_delta_inf: float
    factor_seconds: float
    dense_bytes: int
    dense_chunk_batch_size: int
    dense_chunk_batch_size_env: str | None
    default_adjoint_cap_bytes: int
    default_cap_refuses_661: bool
    ift_stab: float
    newton_stab: float
    ift_used_newton_stab: bool
    wall_seconds: float
    spectrum_unregularized: dict[str, object]
    spectrum_stabilized_from_unregularized: dict[str, object]
    unregularized_positive_definite: bool
    lambda_shift_l2: float | None
    adjoint_live_residual_l2: float | None
    adjoint_live_eta: float | None
    adjoint_dense_eta: float | None
    mixed_scan_index: int | None
    mixed_scan_norm: float | None
    predicted_step_norm: float | None
    vjp_dot: float | None
    predicted_dot: float | None
    vjp_match: bool | None
    adjoint_api_seconds: float | None
    fd_epsilon: float
    fd_control_success: bool | None
    fd_control_iter: int | None
    fd_control_grad_l2: float | None
    fd_perturbed_success: bool | None
    fd_perturbed_iter: int | None
    fd_perturbed_grad_l2: float | None
    fd_rel_l2: float | None
    fd_max_abs: float | None
    fd_match: bool | None
    fd_seconds: float | None
    coil_delta_inf_after: float
    fail_closed_reason: str | None
    runtime: dict[str, object]
    provenance: dict[str, object]

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def evaluate_f3_b37_endpoint_adjoint_probe(
    jax_boozer: BoozerSurfaceJAX,
) -> NestedLsEndpointAdjointProbe:
    """Unregularized 661 IFT adjoint at the opt-in dense-LU walk endpoint.

    Lifts the 1 MiB adjoint cap for this call only. Materializes
    ``Ĥ_ss`` at ``stab=0``. Finite-difference reconvergence uses the
    same unregularized dense LU, not Newton ``stab=1e-4``. Does not
    run a walk, cap-2048, or a moving-coil B3 outer.
    """

    if not F3_B37_DENSE_LU_WALK_EVIDENCE.is_file():
        raise FileNotFoundError(str(F3_B37_DENSE_LU_WALK_EVIDENCE))
    archived = json.loads(F3_B37_DENSE_LU_WALK_EVIDENCE.read_text(encoding="utf-8"))
    archived_probe = archived["probe"]
    frozen = np.ascontiguousarray(
        np.asarray(archived_probe["jax_surface_dofs"], dtype=np.float64)
    )
    stored_sha = str(archived_probe["jax_surface_sha256"])
    provenance = nested_ls_receipt_provenance()
    runtime = {
        "hostname": provenance["hostname"],
        "python_executable": provenance["python_executable"],
        "jax_default_backend": provenance["jax_default_backend"],
        "jax_devices": provenance["jax_devices"],
        "jax_platforms_env": provenance["jax_platforms_env"],
        "jax_enable_x64_env": provenance["jax_enable_x64_env"],
        "timestamp_utc": provenance["timestamp_utc"],
    }
    dense_chunk_batch_size = dense_operator_chunk_batch_size()
    dense_chunk_batch_size_env = os.environ.get(
        "SIMSOPT_DENSE_OPERATOR_CHUNK_BATCH_SIZE"
    )
    dense_bytes = schur_dense_operator_bytes(int(frozen.size))
    default_cap = int(NESTED_LS_IMPLICIT_ADJOINT_DEFAULT_DENSE_BYTES)
    default_cap_refuses = dense_bytes > default_cap
    frozen_sha = sha256_float64(frozen)
    empty_spectrum = _empty_dense_spectrum()

    def _empty(
        reason: str,
        *,
        loaded_sha: str,
        reload_match: bool,
        g_value: float,
        iota: float,
        g_const: float,
        coil: float,
        coil_after: float = 0.0,
    ) -> NestedLsEndpointAdjointProbe:
        grad_json = _json_finite(g_value)
        iota_json = _json_finite(iota)
        g_json = _json_finite(g_const)
        coil_json = _json_finite(coil)
        after_json = _json_finite(coil_after)
        return NestedLsEndpointAdjointProbe(
            surface_sha256=frozen_sha,
            reloaded_surface_sha256=loaded_sha,
            reload_sha_match=reload_match,
            reduced_grad_l2=0.0 if grad_json is None else grad_json,
            iota=0.0 if iota_json is None else iota_json,
            G=0.0 if g_json is None else g_json,
            coil_delta_inf=0.0 if coil_json is None else coil_json,
            factor_seconds=0.0,
            dense_bytes=int(dense_bytes),
            dense_chunk_batch_size=dense_chunk_batch_size,
            dense_chunk_batch_size_env=dense_chunk_batch_size_env,
            default_adjoint_cap_bytes=default_cap,
            default_cap_refuses_661=bool(default_cap_refuses),
            ift_stab=float(F3_B37_IFT_STAB),
            newton_stab=float(NESTED_LS_NEWTON_STAB),
            ift_used_newton_stab=False,
            wall_seconds=float(F3_B37_ADJOINT_WALL_SECONDS),
            spectrum_unregularized=dict(empty_spectrum),
            spectrum_stabilized_from_unregularized=dict(empty_spectrum),
            unregularized_positive_definite=False,
            lambda_shift_l2=None,
            adjoint_live_residual_l2=None,
            adjoint_live_eta=None,
            adjoint_dense_eta=None,
            mixed_scan_index=None,
            mixed_scan_norm=None,
            predicted_step_norm=None,
            vjp_dot=None,
            predicted_dot=None,
            vjp_match=None,
            adjoint_api_seconds=None,
            fd_epsilon=float(F3_B37_ADJOINT_FD_EPSILON),
            fd_control_success=None,
            fd_control_iter=None,
            fd_control_grad_l2=None,
            fd_perturbed_success=None,
            fd_perturbed_iter=None,
            fd_perturbed_grad_l2=None,
            fd_rel_l2=None,
            fd_max_abs=None,
            fd_match=None,
            fd_seconds=None,
            coil_delta_inf_after=0.0 if after_json is None else after_json,
            fail_closed_reason=reason,
            runtime=runtime,
            provenance=provenance,
        )

    if frozen.size != 661 or frozen_sha != F3_B37_DENSE_LU_ENDPOINT_SURFACE_SHA256:
        return _empty(
            "surface_sha_mismatch",
            loaded_sha=frozen_sha,
            reload_match=False,
            g_value=0.0,
            iota=0.0,
            g_const=0.0,
            coil=0.0,
        )
    if stored_sha != F3_B37_DENSE_LU_ENDPOINT_SURFACE_SHA256:
        return _empty(
            "surface_sha_mismatch",
            loaded_sha=stored_sha,
            reload_match=False,
            g_value=0.0,
            iota=0.0,
            g_const=0.0,
            coil=0.0,
        )
    jax_boozer.surface.set_dofs(np.zeros_like(frozen))
    jax_boozer.surface.set_dofs(frozen)
    loaded = np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64)
    loaded_sha = sha256_float64(loaded)
    reload_match = loaded_sha == frozen_sha
    coils_before = np.asarray(jax_boozer.biotsavart.x, dtype=np.float64)
    if not reload_match:
        return _empty(
            "reload_sha_mismatch",
            loaded_sha=loaded_sha,
            reload_match=False,
            g_value=0.0,
            iota=F3_B37_DENSE_LU_ENDPOINT_IOTA,
            g_const=F3_B37_DENSE_LU_ENDPOINT_G,
            coil=0.0,
        )
    iota = float(F3_B37_DENSE_LU_ENDPOINT_IOTA)
    g_const = float(F3_B37_DENSE_LU_ENDPOINT_G)
    if abs(_as_float(archived_probe["iota"]) - iota) > 1.0e-15:
        return _empty(
            "iota_mismatch",
            loaded_sha=loaded_sha,
            reload_match=True,
            g_value=0.0,
            iota=_as_float(archived_probe["iota"]),
            g_const=g_const,
            coil=0.0,
        )
    if abs(_as_float(archived_probe["G"]) - g_const) > 1.0e-15:
        return _empty(
            "g_mismatch",
            loaded_sha=loaded_sha,
            reload_match=True,
            g_value=0.0,
            iota=iota,
            g_const=_as_float(archived_probe["G"]),
            coil=0.0,
        )
    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(jax_boozer)
    del _phi_hat
    y = np.array([iota, g_const], dtype=np.float64)
    y_star = solve_projected_y(residual_fn, loaded, y).solution
    coil = float(
        np.linalg.norm(
            np.asarray(jax_boozer.biotsavart.x, dtype=np.float64) - coils_before,
            ord=np.inf,
        )
    )
    gradient = np.asarray(
        reduced_penalty_gradient_envelope(objective_fn, loaded, y_star),
        dtype=np.float64,
    )
    g_value = float(np.linalg.norm(gradient))
    if (
        not np.isfinite(g_value)
        or abs(g_value - F3_B37_DENSE_LU_ENDPOINT_GRAD_L2) > 1.0e-12
    ):
        return _empty(
            "grad_mismatch",
            loaded_sha=loaded_sha,
            reload_match=True,
            g_value=g_value,
            iota=iota,
            g_const=g_const,
            coil=coil,
        )

    spectrum_unreg = dict(empty_spectrum)
    spectrum_stab = dict(empty_spectrum)
    unregularized_pd = False
    lambda_shift_l2: float | None = None
    live_residual_l2: float | None = None
    live_eta: float | None = None
    dense_eta: float | None = None
    mixed_index: int | None = None
    mixed_norm: float | None = None
    predicted_norm: float | None = None
    vjp_dot: float | None = None
    predicted_dot: float | None = None
    vjp_match: bool | None = None
    adjoint_api_seconds: float | None = None
    fd_control_success: bool | None = None
    fd_control_iter: int | None = None
    fd_control_grad_l2: float | None = None
    fd_perturbed_success: bool | None = None
    fd_perturbed_iter: int | None = None
    fd_perturbed_grad_l2: float | None = None
    fd_rel_l2: float | None = None
    fd_max_abs: float | None = None
    fd_match: bool | None = None
    fd_seconds: float | None = None
    fail_reason: str | None = None
    factor_seconds = 0.0
    coil_after = coil
    deadline = time.perf_counter() + float(F3_B37_ADJOINT_WALL_SECONDS)

    def _surface_sha() -> str:
        return sha256_float64(np.asarray(jax_boozer.surface.get_dofs()))

    if _surface_sha() != frozen_sha:
        fail_reason = "state_digest_drift"
    else:
        factor_started = time.perf_counter()
        operator = factor_reduced_nested_ls_schur(
            residual_fn,
            objective_fn,
            loaded,
            y_probe=np.asarray(y_star, dtype=np.float64),
        )
        factor_seconds = time.perf_counter() - factor_started

        def live_matvec(tangent: jax.Array) -> jax.Array:
            return operator.apply(tangent)

        cotangent_np = np.zeros((int(frozen.size),), dtype=np.float64)
        cotangent_np[0] = 1.0
        cotangent = jnp.asarray(cotangent_np, dtype=jnp.float64)
        dense_started = time.perf_counter()
        materialized: jax.Array = jax.block_until_ready(
            materialize_stabilized_schur_dense(
                operator,
                float(F3_B37_IFT_STAB),
                max_dense_linearization_bytes=None,
            )
        )
        dense_seconds = float(time.perf_counter() - dense_started)
        print(f"adjoint dense assemble s={dense_seconds:.3f}", flush=True)
        dense0_np = np.asarray(jax.device_get(materialized), dtype=np.float64)
        if not np.all(np.isfinite(dense0_np)):
            fail_reason = "nonfinite"
        else:
            lu_started = time.perf_counter()
            lambda0 = jax.block_until_ready(
                solve_stabilized_schur_dense_lu(materialized, cotangent)
            )
            lu_seconds = float(time.perf_counter() - lu_started)
            lambda0_np = np.asarray(jax.device_get(lambda0), dtype=np.float64)
            live_residual_l2, live_eta = _live_unpreconditioned_eta(
                live_matvec, lambda0, cotangent
            )
            dense_residual = jax.block_until_ready(materialized @ lambda0 - cotangent)
            dense_residual_l2 = float(
                np.linalg.norm(
                    np.asarray(jax.device_get(dense_residual), dtype=np.float64)
                )
            )
            rhs_l2 = float(np.linalg.norm(cotangent_np))
            dense_eta = dense_residual_l2 / rhs_l2 if rhs_l2 > 0.0 else 0.0
            spectrum_unreg = _dense_schur_spectrum(dense0_np, cotangent_np, lambda0_np)
            n = int(dense0_np.shape[0])
            dense_stab_np = dense0_np + float(NESTED_LS_NEWTON_STAB) * np.eye(
                n, dtype=np.float64
            )
            lambda_stab_np = np.linalg.solve(dense_stab_np, cotangent_np)
            spectrum_stab = _dense_schur_spectrum(
                dense_stab_np, cotangent_np, lambda_stab_np
            )
            lambda_shift_l2 = _json_finite(
                float(np.linalg.norm(lambda0_np - lambda_stab_np))
            )
            n_negative = _as_int(spectrum_unreg["n_eig_negative_real"])
            eig_abs_min = spectrum_unreg["eig_abs_min"]
            eig_real_min = spectrum_unreg["eig_real_min"]
            unregularized_pd = bool(
                n_negative == 0
                and eig_real_min is not None
                and _as_float(eig_real_min) > 0.0
            )
            print(
                "adjoint spectrum n_neg="
                f"{n_negative} eig_real_min={eig_real_min!r} "
                f"live_eta={live_eta!r} lu_s={lu_seconds:.3f} "
                f"shift={lambda_shift_l2!r}",
                flush=True,
            )
            if _surface_sha() != frozen_sha:
                fail_reason = "state_digest_drift"
            elif (
                not np.all(np.isfinite(lambda0_np))
                or live_eta is None
                or not np.isfinite(live_eta)
            ):
                fail_reason = "nonfinite"
            elif eig_abs_min is None or _as_float(eig_abs_min) <= 0.0:
                fail_reason = "singular_unregularized_hss"
            elif n_negative > 0:
                fail_reason = "indefinite_unregularized_hss"
            elif float(live_eta) > float(F3_B37_ADJOINT_LIVE_ETA_TOL):
                fail_reason = "adjoint_residual_unmet"

            if fail_reason is None and time.perf_counter() < deadline:
                residual_rt, objective_rt, _phi_rt = nested_ls_runtime_coil_closures(
                    jax_boozer
                )
                del _phi_rt
                coil0 = np.array(coils_before, dtype=np.float64, copy=True)
                tangent = None
                mixed_dc = None
                best_norm = -1.0
                best_index = -1
                scan_n = min(int(F3_B37_ADJOINT_COIL_SCAN), int(coil0.size))
                for index in range(scan_n):
                    candidate = np.zeros_like(coil0)
                    candidate[index] = 1.0
                    mixed_candidate = np.asarray(
                        apply_reduced_mixed_schur_coil_tangent(
                            residual_rt,
                            objective_rt,
                            loaded,
                            coil0,
                            candidate,
                            operator=operator,
                        ),
                        dtype=np.float64,
                    )
                    candidate_norm = float(np.linalg.norm(mixed_candidate))
                    print(
                        f"adjoint mixed index={index} norm={candidate_norm!r}",
                        flush=True,
                    )
                    if candidate_norm > best_norm:
                        best_norm = candidate_norm
                        best_index = index
                        tangent = candidate
                        mixed_dc = mixed_candidate
                mixed_index = int(best_index) if best_index >= 0 else None
                mixed_norm = _json_finite(best_norm) if best_norm >= 0.0 else None
                if (
                    tangent is None
                    or mixed_dc is None
                    or best_norm <= float(F3_B37_ADJOINT_MIXED_NORM_FLOOR)
                ):
                    if fail_reason is None:
                        fail_reason = "mixed_scan_dead_zero"
                else:
                    predicted_step = np.asarray(
                        -solve_stabilized_schur_dense_lu(
                            materialized, jnp.asarray(mixed_dc, dtype=jnp.float64)
                        ),
                        dtype=np.float64,
                    )
                    predicted_norm = _json_finite(float(np.linalg.norm(predicted_step)))
                    predicted_dot = _json_finite(-float(np.dot(lambda0_np, mixed_dc)))
                    api_started = time.perf_counter()
                    adjoint = np.asarray(
                        implicit_adjoint_coil_gradient(
                            residual_rt,
                            objective_rt,
                            loaded,
                            coil0,
                            cotangent_np,
                            stab=float(F3_B37_IFT_STAB),
                            linear_solver="dense_lu",
                            operator=operator,
                            max_dense_linearization_bytes=None,
                        ),
                        dtype=np.float64,
                    )
                    adjoint_api_seconds = float(time.perf_counter() - api_started)
                    vjp_dot = _json_finite(float(np.dot(adjoint, tangent)))
                    vjp_ok = bool(
                        vjp_dot is not None
                        and predicted_dot is not None
                        and np.isfinite(vjp_dot)
                        and np.isfinite(predicted_dot)
                        and bool(
                            np.allclose(
                                vjp_dot,
                                predicted_dot,
                                rtol=float(F3_B37_ADJOINT_VJP_RTOL),
                                atol=float(F3_B37_ADJOINT_VJP_ATOL),
                            )
                        )
                    )
                    vjp_match = vjp_ok
                    print(
                        "adjoint vjp "
                        f"dot={vjp_dot!r} predicted={predicted_dot!r} "
                        f"match={vjp_ok} api_s={adjoint_api_seconds:.3f} "
                        f"pred_norm={predicted_norm!r}",
                        flush=True,
                    )
                    if fail_reason is None and not vjp_ok:
                        fail_reason = "adjoint_vjp_mismatch"
                    run_fd = (
                        fail_reason is None
                        and predicted_norm is not None
                        and float(predicted_norm)
                        > float(F3_B37_ADJOINT_MIXED_NORM_FLOOR)
                        and time.perf_counter() < deadline
                    )
                    if run_fd:
                        epsilon = float(F3_B37_ADJOINT_FD_EPSILON)
                        fd_started = time.perf_counter()
                        try:
                            jax_boozer.surface.set_dofs(loaded)
                            jax_boozer.biotsavart.x = coil0
                            jax_boozer._refresh_coil_data()
                            control = run_reduced_nested_ls_schur_newton(
                                jax_boozer,
                                iota=iota,
                                G=g_const,
                                stab=float(F3_B37_IFT_STAB),
                                maxiter=int(NESTED_LS_NEWTON_MAXITER),
                                linear_solver="dense_lu",
                            )
                            fd_control_success = bool(control.success)
                            fd_control_iter = int(control.iteration_count)
                            fd_control_grad_l2 = _json_finite(
                                float(np.linalg.norm(control.reduced_gradient))
                            )
                            print(
                                "adjoint fd control "
                                f"success={control.success} iter={control.iteration_count} "
                                f"g={fd_control_grad_l2!r} coil={control.coil_delta_inf!r}",
                                flush=True,
                            )
                            jax_boozer.surface.set_dofs(loaded)
                            jax_boozer.biotsavart.x = coil0 + epsilon * tangent
                            jax_boozer._refresh_coil_data()
                            perturbed = run_reduced_nested_ls_schur_newton(
                                jax_boozer,
                                iota=iota,
                                G=g_const,
                                stab=float(F3_B37_IFT_STAB),
                                maxiter=int(NESTED_LS_NEWTON_MAXITER),
                                linear_solver="dense_lu",
                            )
                            fd_perturbed_success = bool(perturbed.success)
                            fd_perturbed_iter = int(perturbed.iteration_count)
                            fd_perturbed_grad_l2 = _json_finite(
                                float(np.linalg.norm(perturbed.reduced_gradient))
                            )
                            finite_difference = (
                                perturbed.surface_dofs - control.surface_dofs
                            ) / epsilon
                            delta = predicted_step - finite_difference
                            pred_scale = float(np.linalg.norm(predicted_step))
                            fd_rel_l2 = _json_finite(
                                float(np.linalg.norm(delta) / pred_scale)
                                if pred_scale > 0.0
                                else float(np.linalg.norm(delta))
                            )
                            fd_max_abs = _json_finite(float(np.max(np.abs(delta))))
                            fd_atol = max(
                                float(F3_B37_ADJOINT_FD_ATOL_FLOOR),
                                float(F3_B37_ADJOINT_FD_ATOL_REL) * pred_scale,
                            )
                            fd_match = bool(
                                fd_control_success
                                and fd_perturbed_success
                                and control.coil_delta_inf == 0.0
                                and perturbed.coil_delta_inf == 0.0
                                and np.allclose(
                                    predicted_step,
                                    finite_difference,
                                    rtol=float(F3_B37_ADJOINT_FD_RTOL),
                                    atol=fd_atol,
                                )
                            )
                            fd_seconds = float(time.perf_counter() - fd_started)
                            print(
                                "adjoint fd perturbed "
                                f"success={perturbed.success} iter={perturbed.iteration_count} "
                                f"g={fd_perturbed_grad_l2!r} rel={fd_rel_l2!r} "
                                f"match={fd_match} s={fd_seconds:.3f}",
                                flush=True,
                            )
                            if fail_reason is None and not bool(fd_control_success):
                                fail_reason = "fd_control_failed"
                            elif fail_reason is None and not bool(fd_perturbed_success):
                                fail_reason = "fd_perturbed_failed"
                            elif fail_reason is None and (
                                control.coil_delta_inf != 0.0
                                or perturbed.coil_delta_inf != 0.0
                            ):
                                fail_reason = "coil_moved"
                            elif fail_reason is None and not bool(fd_match):
                                fail_reason = "fd_mismatch"
                        finally:
                            jax_boozer.biotsavart.x = coil0
                            jax_boozer._refresh_coil_data()
                            jax_boozer.surface.set_dofs(loaded)
                    elif fail_reason is None and time.perf_counter() >= deadline:
                        fail_reason = "wall_time_cap"
            elif fail_reason is None and time.perf_counter() >= deadline:
                fail_reason = "wall_time_cap"

    coil_after = float(
        np.linalg.norm(
            np.asarray(jax_boozer.biotsavart.x, dtype=np.float64) - coils_before,
            ord=np.inf,
        )
    )
    if fail_reason is None and coil_after != 0.0:
        fail_reason = "coil_moved"
    if fail_reason is None and _surface_sha() != frozen_sha:
        fail_reason = "state_digest_drift"

    return NestedLsEndpointAdjointProbe(
        surface_sha256=frozen_sha,
        reloaded_surface_sha256=loaded_sha,
        reload_sha_match=reload_match,
        reduced_grad_l2=g_value,
        iota=iota,
        G=g_const,
        coil_delta_inf=coil,
        factor_seconds=float(factor_seconds),
        dense_bytes=int(dense_bytes),
        dense_chunk_batch_size=dense_chunk_batch_size,
        dense_chunk_batch_size_env=dense_chunk_batch_size_env,
        default_adjoint_cap_bytes=default_cap,
        default_cap_refuses_661=bool(default_cap_refuses),
        ift_stab=float(F3_B37_IFT_STAB),
        newton_stab=float(NESTED_LS_NEWTON_STAB),
        ift_used_newton_stab=False,
        wall_seconds=float(F3_B37_ADJOINT_WALL_SECONDS),
        spectrum_unregularized=spectrum_unreg,
        spectrum_stabilized_from_unregularized=spectrum_stab,
        unregularized_positive_definite=bool(unregularized_pd),
        lambda_shift_l2=lambda_shift_l2,
        adjoint_live_residual_l2=_json_finite(live_residual_l2)
        if live_residual_l2 is not None
        else None,
        adjoint_live_eta=_json_finite(live_eta) if live_eta is not None else None,
        adjoint_dense_eta=_json_finite(dense_eta) if dense_eta is not None else None,
        mixed_scan_index=mixed_index,
        mixed_scan_norm=mixed_norm,
        predicted_step_norm=predicted_norm,
        vjp_dot=vjp_dot,
        predicted_dot=predicted_dot,
        vjp_match=vjp_match,
        adjoint_api_seconds=adjoint_api_seconds,
        fd_epsilon=float(F3_B37_ADJOINT_FD_EPSILON),
        fd_control_success=fd_control_success,
        fd_control_iter=fd_control_iter,
        fd_control_grad_l2=fd_control_grad_l2,
        fd_perturbed_success=fd_perturbed_success,
        fd_perturbed_iter=fd_perturbed_iter,
        fd_perturbed_grad_l2=fd_perturbed_grad_l2,
        fd_rel_l2=fd_rel_l2,
        fd_max_abs=fd_max_abs,
        fd_match=fd_match,
        fd_seconds=fd_seconds,
        coil_delta_inf_after=coil_after,
        fail_closed_reason=fail_reason,
        runtime=runtime,
        provenance=provenance,
    )


def _freeze_dense_lu_walk_endpoint(
    jax_boozer: BoozerSurfaceJAX,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
    float,
    str | None,
]:
    """Load the dense-LU walk endpoint onto ``jax_boozer``. Coils stay frozen."""

    if not F3_B37_DENSE_LU_WALK_EVIDENCE.is_file():
        raise FileNotFoundError(str(F3_B37_DENSE_LU_WALK_EVIDENCE))
    archived = json.loads(F3_B37_DENSE_LU_WALK_EVIDENCE.read_text(encoding="utf-8"))
    archived_probe = archived["probe"]
    frozen = np.ascontiguousarray(
        np.asarray(archived_probe["jax_surface_dofs"], dtype=np.float64)
    )
    stored_sha = str(archived_probe["jax_surface_sha256"])
    frozen_sha = sha256_float64(frozen)
    coils_before = np.asarray(jax_boozer.biotsavart.x, dtype=np.float64)
    empty = (
        frozen,
        coils_before,
        np.zeros(2, dtype=np.float64),
        0.0,
        0.0,
        0.0,
    )
    if frozen.size != 661 or frozen_sha != F3_B37_DENSE_LU_ENDPOINT_SURFACE_SHA256:
        return (*empty, "surface_sha_mismatch")
    if stored_sha != F3_B37_DENSE_LU_ENDPOINT_SURFACE_SHA256:
        return (*empty, "surface_sha_mismatch")
    iota = float(F3_B37_DENSE_LU_ENDPOINT_IOTA)
    g_const = float(F3_B37_DENSE_LU_ENDPOINT_G)
    if abs(_as_float(archived_probe["iota"]) - iota) > 1.0e-15:
        return (*empty, "iota_mismatch")
    if abs(_as_float(archived_probe["G"]) - g_const) > 1.0e-15:
        return (*empty, "g_mismatch")
    jax_boozer.surface.set_dofs(np.zeros_like(frozen))
    jax_boozer.surface.set_dofs(frozen)
    loaded = np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64)
    if sha256_float64(loaded) != frozen_sha:
        return (*empty, "reload_sha_mismatch")
    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(jax_boozer)
    del _phi_hat
    y_star = solve_projected_y(
        residual_fn, loaded, np.array([iota, g_const], dtype=np.float64)
    ).solution
    gradient = np.asarray(
        reduced_penalty_gradient_envelope(objective_fn, loaded, y_star),
        dtype=np.float64,
    )
    g_value = float(np.linalg.norm(gradient))
    coil = float(
        np.linalg.norm(
            np.asarray(jax_boozer.biotsavart.x, dtype=np.float64) - coils_before,
            ord=np.inf,
        )
    )
    if coil != 0.0:
        return (
            loaded,
            coils_before,
            np.asarray(y_star),
            iota,
            g_const,
            g_value,
            "coil_moved",
        )
    if (
        not np.isfinite(g_value)
        or abs(g_value - F3_B37_DENSE_LU_ENDPOINT_GRAD_L2) > 1.0e-12
    ):
        return (
            loaded,
            coils_before,
            np.asarray(y_star),
            iota,
            g_const,
            g_value,
            "grad_mismatch",
        )
    return (
        loaded,
        coils_before,
        np.asarray(y_star, dtype=np.float64),
        iota,
        g_const,
        g_value,
        None,
    )


@dataclass(frozen=True, slots=True)
class NestedLsChunkBananaProbe:
    """Native banana ``run_code`` plus dense-assemble chunk sweep.

    Reconstruct Newton is a different operator; ``comparable_operators``
    is false. Not a nested speed claim and not F3 7.70×.
    """

    banana_success: bool
    banana_iter: int
    banana_newton_iter: int
    banana_bfgs_iter: int | None
    banana_iter_meaning: str
    banana_iota: float
    banana_G: float
    banana_seconds: float
    banana_bfgs_seconds: float | None
    banana_newton_seconds: float | None
    banana_coil_delta_inf: float
    banana_omp_pinned: bool
    banana_omp_num_threads: str | None
    reconstruct_walk_seconds: float | None
    reconstruct_walk_success: bool | None
    y_star_iota: float
    y_star_g: float
    endpoint_sha256: str
    newton_stab: float
    default_chunk_batch_size: int
    rows: tuple[dict[str, object], ...]
    matrix_rel_frobenius_max: float | None
    fail_closed_reason: str | None
    runtime: dict[str, object]
    provenance: dict[str, object]

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def evaluate_f3_b37_chunk_banana_probe(
    native: BoozerSurface,
    jax_boozer: BoozerSurfaceJAX,
) -> NestedLsChunkBananaProbe:
    """Native banana bar from F3 B37 start; chunk sweep at the LU endpoint.

    Does not rerun the reconstruct walk. Does not switch the Newton
    default. Chunk widths are opt-in ``chunk_batch_size`` arguments.
    """

    provenance = nested_ls_receipt_provenance()
    runtime = {
        "hostname": provenance["hostname"],
        "python_executable": provenance["python_executable"],
        "jax_default_backend": provenance["jax_default_backend"],
        "jax_devices": provenance["jax_devices"],
        "jax_platforms_env": provenance["jax_platforms_env"],
        "jax_enable_x64_env": provenance["jax_enable_x64_env"],
        "timestamp_utc": provenance["timestamp_utc"],
    }
    default_chunk = dense_operator_chunk_batch_size()
    walk_seconds: float | None = None
    walk_success: bool | None = None
    if F3_B37_DENSE_LU_WALK_EVIDENCE.is_file():
        walk_payload = json.loads(
            F3_B37_DENSE_LU_WALK_EVIDENCE.read_text(encoding="utf-8")
        )
        walk_probe = walk_payload["probe"]
        walk_seconds = _json_finite(_as_float(walk_probe["walk_seconds"]))
        walk_success = bool(walk_probe["success"])

    def _empty(
        reason: str,
        *,
        banana_success: bool = False,
        banana_iter: int = 0,
        banana_iota: float = 0.0,
        banana_g: float = 0.0,
        banana_seconds: float = 0.0,
        banana_coil: float = 0.0,
        y_iota: float = 0.0,
        y_g: float = 0.0,
        sha: str = "",
        rows: tuple[dict[str, object], ...] = (),
        matrix_rel: float | None = None,
    ) -> NestedLsChunkBananaProbe:
        return NestedLsChunkBananaProbe(
            banana_success=banana_success,
            banana_iter=banana_iter,
            banana_newton_iter=banana_iter,
            banana_bfgs_iter=None,
            banana_iter_meaning=(
                "run_code returns Newton polish nit after BFGS; "
                "seconds are BFGS+Newton wall"
            ),
            banana_iota=banana_iota,
            banana_G=banana_g,
            banana_seconds=banana_seconds,
            banana_bfgs_seconds=None,
            banana_newton_seconds=None,
            banana_coil_delta_inf=banana_coil,
            banana_omp_pinned=nested_ls_omp_threads_pinned(),
            banana_omp_num_threads=nested_ls_threading_env()["OMP_NUM_THREADS"],
            reconstruct_walk_seconds=walk_seconds,
            reconstruct_walk_success=walk_success,
            y_star_iota=y_iota,
            y_star_g=y_g,
            endpoint_sha256=sha,
            newton_stab=float(NESTED_LS_NEWTON_STAB),
            default_chunk_batch_size=default_chunk,
            rows=rows,
            matrix_rel_frobenius_max=matrix_rel,
            fail_closed_reason=reason,
            runtime=runtime,
            provenance=provenance,
        )

    residual_fn, _objective_fn, _phi_hat = nested_ls_reduced_closures(jax_boozer)
    del _phi_hat
    surface0 = np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64)
    native_start = np.asarray(native.surface.get_dofs(), dtype=np.float64)
    solution = solve_projected_y(residual_fn, surface0, np.zeros(2, dtype=np.float64))
    require_full_y_rank(solution)
    y_star = np.asarray(solution.solution, dtype=np.float64)
    banana_options = nested_ls_banana_run_code_options()
    original_options = replace_native_solver_options(
        native,
        {
            "newton_tol": banana_options["newton_tol"],
            "newton_maxiter": banana_options["newton_maxiter"],
            "bfgs_tol": banana_options["bfgs_tol"],
        },
    )
    native.surface.set_dofs(native_start)
    native_coils0 = np.asarray(native.biotsavart.x, dtype=np.float64)
    native.need_to_run_code = True
    banana_started = time.perf_counter()
    try:
        banana = native.run_code(float(y_star[0]), G=float(y_star[1]))
        banana_seconds = time.perf_counter() - banana_started
    finally:
        native.options = original_options
    if banana is None:
        return _empty("banana_none", y_iota=float(y_star[0]), y_g=float(y_star[1]))
    banana_coil = float(
        np.linalg.norm(
            np.asarray(native.biotsavart.x, dtype=np.float64) - native_coils0,
            ord=np.inf,
        )
    )
    banana_success = bool(banana["success"])
    banana_iter = int(banana["iter"])
    banana_iota = float(banana["iota"])
    banana_g = float(banana["G"])
    banana_omp_pinned = nested_ls_omp_threads_pinned()
    banana_omp_threads = nested_ls_threading_env()["OMP_NUM_THREADS"]
    print(
        "banana native"
        f" success={banana_success} newton_iter={banana_iter}"
        f" s={banana_seconds:.3f} coil={banana_coil!r}"
        f" omp={banana_omp_threads!r} pinned={banana_omp_pinned}",
        flush=True,
    )
    loaded, coils, y_end, _iota, _g_const, _g_value, freeze_reason = (
        _freeze_dense_lu_walk_endpoint(jax_boozer)
    )
    if freeze_reason is not None:
        return _empty(
            freeze_reason,
            banana_success=banana_success,
            banana_iter=banana_iter,
            banana_iota=banana_iota,
            banana_g=banana_g,
            banana_seconds=float(banana_seconds),
            banana_coil=banana_coil,
            y_iota=float(y_star[0]),
            y_g=float(y_star[1]),
            sha=sha256_float64(loaded),
        )
    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(jax_boozer)
    del _phi_hat
    operator = factor_reduced_nested_ls_schur(
        residual_fn,
        objective_fn,
        loaded,
        y_probe=np.asarray(y_end, dtype=np.float64),
    )
    reference: np.ndarray | None = None
    rel_max = 0.0
    rows: list[dict[str, object]] = []
    deadline = time.perf_counter() + float(F3_B37_CHUNK_BANANA_WALL_SECONDS)
    fail_reason: str | None = None
    for width in F3_B37_CHUNK_WIDTHS:
        if time.perf_counter() >= deadline:
            fail_reason = "wall_time_cap"
            break
        started = time.perf_counter()
        dense = jax.block_until_ready(
            materialize_stabilized_schur_dense(
                operator,
                float(NESTED_LS_NEWTON_STAB),
                chunk_batch_size=int(width),
            )
        )
        dense_seconds = float(time.perf_counter() - started)
        dense_np = np.asarray(jax.device_get(dense), dtype=np.float64)
        if not np.all(np.isfinite(dense_np)):
            fail_reason = "nonfinite"
            break
        lu_seconds = 0.0
        live_eta: float | None = None
        if reference is None:
            lu_started = time.perf_counter()
            rhs = jnp.ones((int(loaded.size),), dtype=jnp.float64)
            delta = jax.block_until_ready(solve_stabilized_schur_dense_lu(dense, rhs))
            lu_seconds = float(time.perf_counter() - lu_started)
            residual = jax.block_until_ready(
                operator.apply(delta) + float(NESTED_LS_NEWTON_STAB) * delta - rhs
            )
            residual_l2 = float(
                np.linalg.norm(np.asarray(jax.device_get(residual), dtype=np.float64))
            )
            rhs_l2 = float(np.linalg.norm(np.ones(int(loaded.size), dtype=np.float64)))
            live_eta = residual_l2 / rhs_l2 if rhs_l2 > 0.0 else 0.0
            reference = dense_np
            rel = 0.0
        else:
            scale = float(np.linalg.norm(reference, ord="fro"))
            rel = (
                float(np.linalg.norm(dense_np - reference, ord="fro") / scale)
                if scale > 0.0
                else float(np.linalg.norm(dense_np - reference, ord="fro"))
            )
            rel_max = max(rel_max, rel)
        print(
            f"chunk width={width} dense_s={dense_seconds:.3f} lu_s={lu_seconds:.3f}"
            f" rel={rel!r} eta={live_eta!r}",
            flush=True,
        )
        rows.append(
            {
                "chunk_batch_size": int(width),
                "dense_seconds": dense_seconds,
                "lu_seconds": lu_seconds,
                "live_eta": _json_finite(live_eta) if live_eta is not None else None,
                "matrix_rel_frobenius": _json_finite(rel),
                "dense_bytes": schur_dense_operator_bytes(int(loaded.size)),
            }
        )
    coil_after = float(
        np.linalg.norm(
            np.asarray(jax_boozer.biotsavart.x, dtype=np.float64) - coils,
            ord=np.inf,
        )
    )
    if fail_reason is None and coil_after != 0.0:
        fail_reason = "coil_moved"
    if fail_reason is None and banana_coil != 0.0:
        fail_reason = "coil_moved"
    return NestedLsChunkBananaProbe(
        banana_success=banana_success,
        banana_iter=banana_iter,
        banana_newton_iter=banana_iter,
        banana_bfgs_iter=None,
        banana_iter_meaning=(
            "run_code returns Newton polish nit after BFGS; "
            "seconds are BFGS+Newton wall"
        ),
        banana_iota=banana_iota,
        banana_G=banana_g,
        banana_seconds=float(banana_seconds),
        banana_bfgs_seconds=None,
        banana_newton_seconds=None,
        banana_coil_delta_inf=banana_coil,
        banana_omp_pinned=banana_omp_pinned,
        banana_omp_num_threads=banana_omp_threads,
        reconstruct_walk_seconds=walk_seconds,
        reconstruct_walk_success=walk_success,
        y_star_iota=float(y_star[0]),
        y_star_g=float(y_star[1]),
        endpoint_sha256=sha256_float64(loaded),
        newton_stab=float(NESTED_LS_NEWTON_STAB),
        default_chunk_batch_size=default_chunk,
        rows=tuple(rows),
        matrix_rel_frobenius_max=_json_finite(rel_max),
        fail_closed_reason=fail_reason,
        runtime=runtime,
        provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class NestedLsVolumeOuterProbe:
    """Moving-coil Volume outer gradient at the dense-LU endpoint.

    Cotangent is ``∇_s Volume``. Inner reconvergence is unregularized
    dense LU. One coil-direction FD. Not an outer optimizer loop.
    """

    surface_sha256: str
    volume: float
    volume_grad_l2: float
    iota: float
    G: float
    reduced_grad_l2: float
    mixed_scan_index: int | None
    mixed_scan_norm: float | None
    predicted_dot: float | None
    fd_dot: float | None
    fd_rel: float | None
    fd_match: bool | None
    vjp_match: bool | None
    adjoint_live_eta: float | None
    fd_control_success: bool | None
    fd_perturbed_success: bool | None
    fd_control_iter: int | None
    fd_perturbed_iter: int | None
    coil_delta_inf: float
    coil_delta_inf_after: float
    fail_closed_reason: str | None
    runtime: dict[str, object]
    provenance: dict[str, object]

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def evaluate_f3_b37_volume_outer_probe(
    jax_boozer: BoozerSurfaceJAX,
) -> NestedLsVolumeOuterProbe:
    """Volume IFT outer gradient + one-direction coil FD at the LU endpoint."""

    provenance = nested_ls_receipt_provenance()
    runtime = {
        "hostname": provenance["hostname"],
        "python_executable": provenance["python_executable"],
        "jax_default_backend": provenance["jax_default_backend"],
        "jax_devices": provenance["jax_devices"],
        "jax_platforms_env": provenance["jax_platforms_env"],
        "jax_enable_x64_env": provenance["jax_enable_x64_env"],
        "timestamp_utc": provenance["timestamp_utc"],
    }

    def _empty(
        reason: str,
        *,
        sha: str = "",
        volume: float = 0.0,
        volume_grad: float = 0.0,
        iota: float = 0.0,
        g_const: float = 0.0,
        g_value: float = 0.0,
        coil: float = 0.0,
        coil_after: float = 0.0,
    ) -> NestedLsVolumeOuterProbe:
        return NestedLsVolumeOuterProbe(
            surface_sha256=sha,
            volume=volume,
            volume_grad_l2=volume_grad,
            iota=iota,
            G=g_const,
            reduced_grad_l2=g_value,
            mixed_scan_index=None,
            mixed_scan_norm=None,
            predicted_dot=None,
            fd_dot=None,
            fd_rel=None,
            fd_match=None,
            vjp_match=None,
            adjoint_live_eta=None,
            fd_control_success=None,
            fd_perturbed_success=None,
            fd_control_iter=None,
            fd_perturbed_iter=None,
            coil_delta_inf=coil,
            coil_delta_inf_after=coil_after,
            fail_closed_reason=reason,
            runtime=runtime,
            provenance=provenance,
        )

    loaded, coils, y_star, iota, g_const, g_value, freeze_reason = (
        _freeze_dense_lu_walk_endpoint(jax_boozer)
    )
    sha = sha256_float64(loaded)
    if freeze_reason is not None:
        return _empty(
            freeze_reason,
            sha=sha,
            iota=iota,
            g_const=g_const,
            g_value=g_value,
        )
    label = jax_boozer.label
    if not isinstance(label, Volume):
        return _empty(
            "label_not_volume",
            sha=sha,
            iota=iota,
            g_const=g_const,
            g_value=g_value,
        )
    volume0 = float(label.J())
    volume_grad = np.asarray(label.dJ_by_dsurfacecoefficients(), dtype=np.float64)
    volume_grad_l2 = float(np.linalg.norm(volume_grad))
    if not np.isfinite(volume0) or volume_grad_l2 <= 0.0:
        return _empty(
            "volume_grad_dead",
            sha=sha,
            volume=volume0,
            volume_grad=volume_grad_l2,
            iota=iota,
            g_const=g_const,
            g_value=g_value,
        )
    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(jax_boozer)
    del _phi_hat
    operator = factor_reduced_nested_ls_schur(
        residual_fn,
        objective_fn,
        loaded,
        y_probe=np.asarray(y_star, dtype=np.float64),
    )
    cotangent = jnp.asarray(volume_grad, dtype=jnp.float64)
    dense = jax.block_until_ready(
        materialize_stabilized_schur_dense(
            operator,
            float(F3_B37_IFT_STAB),
            max_dense_linearization_bytes=None,
        )
    )
    lambda0 = jax.block_until_ready(solve_stabilized_schur_dense_lu(dense, cotangent))
    residual = jax.block_until_ready(operator.apply(lambda0) - cotangent)
    residual_l2 = float(
        np.linalg.norm(np.asarray(jax.device_get(residual), dtype=np.float64))
    )
    rhs_l2 = float(np.linalg.norm(volume_grad))
    live_eta = residual_l2 / rhs_l2 if rhs_l2 > 0.0 else 0.0
    print(f"volume outer live_eta={live_eta!r} V={volume0!r}", flush=True)
    fail_reason: str | None = None
    if not np.isfinite(live_eta):
        fail_reason = "nonfinite"
    elif live_eta > float(F3_B37_ADJOINT_LIVE_ETA_TOL):
        fail_reason = "adjoint_residual_unmet"
    mixed_index: int | None = None
    mixed_norm: float | None = None
    predicted_dot: float | None = None
    fd_dot: float | None = None
    fd_rel: float | None = None
    fd_match: bool | None = None
    vjp_match: bool | None = None
    fd_control_success: bool | None = None
    fd_perturbed_success: bool | None = None
    fd_control_iter: int | None = None
    fd_perturbed_iter: int | None = None
    coil0 = np.array(coils, dtype=np.float64, copy=True)
    if fail_reason is None:
        residual_rt, objective_rt, _phi_rt = nested_ls_runtime_coil_closures(jax_boozer)
        del _phi_rt
        tangent = None
        best_norm = -1.0
        best_index = -1
        scan_n = min(int(F3_B37_ADJOINT_COIL_SCAN), int(coil0.size))
        for index in range(scan_n):
            candidate = np.zeros_like(coil0)
            candidate[index] = 1.0
            mixed_candidate = np.asarray(
                apply_reduced_mixed_schur_coil_tangent(
                    residual_rt,
                    objective_rt,
                    loaded,
                    coil0,
                    candidate,
                    operator=operator,
                ),
                dtype=np.float64,
            )
            candidate_norm = float(np.linalg.norm(mixed_candidate))
            print(f"volume mixed index={index} norm={candidate_norm!r}", flush=True)
            if candidate_norm > best_norm:
                best_norm = candidate_norm
                best_index = index
                tangent = candidate
        mixed_index = int(best_index) if best_index >= 0 else None
        mixed_norm = _json_finite(best_norm) if best_norm >= 0.0 else None
        if tangent is None or best_norm <= float(F3_B37_ADJOINT_MIXED_NORM_FLOOR):
            fail_reason = "mixed_scan_dead_zero"
        else:
            lambda_np = np.asarray(jax.device_get(lambda0), dtype=np.float64)
            adjoint = np.asarray(
                implicit_adjoint_coil_gradient(
                    residual_rt,
                    objective_rt,
                    loaded,
                    coil0,
                    volume_grad,
                    stab=float(F3_B37_IFT_STAB),
                    linear_solver="dense_lu",
                    operator=operator,
                    max_dense_linearization_bytes=None,
                ),
                dtype=np.float64,
            )
            predicted_dot = _json_finite(float(np.dot(adjoint, tangent)))
            mixed_dc = np.asarray(
                apply_reduced_mixed_schur_coil_tangent(
                    residual_rt,
                    objective_rt,
                    loaded,
                    coil0,
                    tangent,
                    operator=operator,
                ),
                dtype=np.float64,
            )
            envelope_dot = _json_finite(-float(np.dot(lambda_np, mixed_dc)))
            vjp_ok = bool(
                predicted_dot is not None
                and envelope_dot is not None
                and np.allclose(
                    predicted_dot,
                    envelope_dot,
                    rtol=float(F3_B37_ADJOINT_VJP_RTOL),
                    atol=float(F3_B37_ADJOINT_VJP_ATOL),
                )
            )
            vjp_match = vjp_ok
            if not vjp_ok:
                fail_reason = "adjoint_vjp_mismatch"
            else:
                epsilon = float(F3_B37_ADJOINT_FD_EPSILON)
                try:
                    jax_boozer.surface.set_dofs(loaded)
                    jax_boozer.biotsavart.x = coil0
                    jax_boozer._refresh_coil_data()
                    control = run_reduced_nested_ls_schur_newton(
                        jax_boozer,
                        iota=iota,
                        G=g_const,
                        stab=float(F3_B37_IFT_STAB),
                        maxiter=int(NESTED_LS_NEWTON_MAXITER),
                        linear_solver="dense_lu",
                    )
                    fd_control_success = bool(control.success)
                    fd_control_iter = int(control.iteration_count)
                    volume_ctrl = float(label.J())
                    jax_boozer.surface.set_dofs(loaded)
                    jax_boozer.biotsavart.x = coil0 + epsilon * tangent
                    jax_boozer._refresh_coil_data()
                    perturbed = run_reduced_nested_ls_schur_newton(
                        jax_boozer,
                        iota=iota,
                        G=g_const,
                        stab=float(F3_B37_IFT_STAB),
                        maxiter=int(NESTED_LS_NEWTON_MAXITER),
                        linear_solver="dense_lu",
                    )
                    fd_perturbed_success = bool(perturbed.success)
                    fd_perturbed_iter = int(perturbed.iteration_count)
                    volume_pert = float(label.J())
                    fd_dot = _json_finite((volume_pert - volume_ctrl) / epsilon)
                    pred_scale = (
                        abs(float(predicted_dot)) if predicted_dot is not None else 0.0
                    )
                    fd_rel = _json_finite(
                        abs(float(predicted_dot) - float(fd_dot)) / pred_scale
                        if predicted_dot is not None
                        and fd_dot is not None
                        and pred_scale > 0.0
                        else float("nan")
                    )
                    fd_atol = max(
                        float(F3_B37_ADJOINT_FD_ATOL_FLOOR),
                        float(F3_B37_ADJOINT_FD_ATOL_REL) * pred_scale,
                    )
                    fd_match = bool(
                        fd_control_success
                        and fd_perturbed_success
                        and control.coil_delta_inf == 0.0
                        and perturbed.coil_delta_inf == 0.0
                        and predicted_dot is not None
                        and fd_dot is not None
                        and np.allclose(
                            predicted_dot,
                            fd_dot,
                            rtol=float(F3_B37_ADJOINT_FD_RTOL),
                            atol=fd_atol,
                        )
                    )
                    print(
                        "volume fd"
                        f" pred={predicted_dot!r} fd={fd_dot!r} rel={fd_rel!r}"
                        f" match={fd_match} ctrl_iter={fd_control_iter}"
                        f" pert_iter={fd_perturbed_iter}",
                        flush=True,
                    )
                    if not bool(fd_control_success):
                        fail_reason = "fd_control_failed"
                    elif not bool(fd_perturbed_success):
                        fail_reason = "fd_perturbed_failed"
                    elif (
                        control.coil_delta_inf != 0.0 or perturbed.coil_delta_inf != 0.0
                    ):
                        fail_reason = "coil_moved"
                    elif not bool(fd_match):
                        fail_reason = "fd_mismatch"
                finally:
                    jax_boozer.biotsavart.x = coil0
                    jax_boozer._refresh_coil_data()
                    jax_boozer.surface.set_dofs(loaded)
    coil_after = float(
        np.linalg.norm(
            np.asarray(jax_boozer.biotsavart.x, dtype=np.float64) - coil0,
            ord=np.inf,
        )
    )
    if fail_reason is None and coil_after != 0.0:
        fail_reason = "coil_moved"
    return NestedLsVolumeOuterProbe(
        surface_sha256=sha,
        volume=volume0,
        volume_grad_l2=volume_grad_l2,
        iota=iota,
        G=g_const,
        reduced_grad_l2=g_value,
        mixed_scan_index=mixed_index,
        mixed_scan_norm=mixed_norm,
        predicted_dot=predicted_dot,
        fd_dot=fd_dot,
        fd_rel=fd_rel,
        fd_match=fd_match,
        vjp_match=vjp_match,
        adjoint_live_eta=_json_finite(live_eta),
        fd_control_success=fd_control_success,
        fd_perturbed_success=fd_perturbed_success,
        fd_control_iter=fd_control_iter,
        fd_perturbed_iter=fd_perturbed_iter,
        coil_delta_inf=0.0,
        coil_delta_inf_after=coil_after,
        fail_closed_reason=fail_reason,
        runtime=runtime,
        provenance=provenance,
    )


def replace_native_solver_options(
    native: BoozerSurface,
    overlay: dict[str, object],
) -> dict[str, object]:
    """Replace ``native.options`` with a copy plus ``overlay``. Return the old dict."""

    original = native.options
    native.options = {**original, **overlay}
    return original


def _run_native_banana_bfgs_then_newton(
    native: BoozerSurface,
    *,
    iota: float,
    G: float,
) -> dict[str, object]:
    """BFGS then Newton as ``run_code`` does, with split walls and OMP env."""

    banana_options = nested_ls_banana_run_code_options()
    original_options = replace_native_solver_options(
        native,
        {
            "newton_tol": banana_options["newton_tol"],
            "newton_maxiter": banana_options["newton_maxiter"],
            "bfgs_tol": banana_options["bfgs_tol"],
        },
    )
    coils0 = np.asarray(native.biotsavart.x, dtype=np.float64)
    threading = nested_ls_threading_env()
    try:
        native.need_to_run_code = True
        bfgs_started = time.perf_counter()
        bfgs = native.minimize_boozer_penalty_constraints_LBFGS(
            constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
            iota=float(iota),
            G=float(G),
            tol=float(banana_options["bfgs_tol"]),
            maxiter=_as_int(native.options.get("bfgs_maxiter", 1500)),
            verbose=False,
            limited_memory=bool(native.options.get("limited_memory", False)),
            weight_inv_modB=NESTED_LS_WEIGHT_INV_MODB,
        )
        bfgs_seconds = float(time.perf_counter() - bfgs_started)
        native.need_to_run_code = True
        newton_started = time.perf_counter()
        newton = native.minimize_boozer_penalty_constraints_newton(
            constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
            iota=float(bfgs["iota"]),
            G=float(bfgs["G"]),
            verbose=False,
            tol=float(banana_options["newton_tol"]),
            maxiter=int(banana_options["newton_maxiter"]),
            weight_inv_modB=NESTED_LS_WEIGHT_INV_MODB,
        )
        newton_seconds = float(time.perf_counter() - newton_started)
    finally:
        native.options = original_options
    coil_delta = float(
        np.linalg.norm(
            np.asarray(native.biotsavart.x, dtype=np.float64) - coils0,
            ord=np.inf,
        )
    )
    return {
        "success": bool(newton["success"]),
        "bfgs_iter": int(bfgs["iter"]),
        "newton_iter": int(newton["iter"]),
        "iter": int(newton["iter"]),
        "iota": float(newton["iota"]),
        "G": float(newton["G"]),
        "bfgs_seconds": bfgs_seconds,
        "newton_seconds": newton_seconds,
        "seconds": bfgs_seconds + newton_seconds,
        "coil_delta_inf": coil_delta,
        "threading": threading,
        "omp_pinned": nested_ls_omp_threads_pinned(threading),
        "omp_num_threads": threading["OMP_NUM_THREADS"],
    }


@dataclass(frozen=True, slots=True)
class NestedLsBananaOmpSweep:
    """OMP-pinned interleaved native banana ``run_code`` sweep.

    Each row is a fresh process with ``OMP_NUM_THREADS`` set before
    import. Not a nested speed claim.
    """

    threads: tuple[int, ...]
    repeats: int
    rows: tuple[dict[str, object], ...]
    any_unpinned: bool
    fail_closed_reason: str | None
    runtime: dict[str, object]
    provenance: dict[str, object]

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def evaluate_f3_b37_banana_omp_sweep(
    *,
    python_executable: str | None = None,
    threads: tuple[int, ...] = F3_B37_BANANA_OMP_THREADS,
    repeats: int = F3_B37_BANANA_OMP_REPEATS,
) -> NestedLsBananaOmpSweep:
    """Launch interleaved native banana children with pinned OpenMP."""

    provenance = nested_ls_receipt_provenance()
    runtime = nested_ls_runtime_identity()
    child = (
        Path(__file__).resolve().parents[3]
        / "benchmarks"
        / "nested_ls_banana_omp_child.py"
    )
    python = python_executable or sys.executable
    rows: list[dict[str, object]] = []
    fail_reason: str | None = None
    any_unpinned = False
    deadline = time.perf_counter() + float(F3_B37_BANANA_OMP_WALL_SECONDS)
    schedule = [
        (repeat, int(thread))
        for repeat in range(int(repeats))
        for thread in threads
    ]
    for repeat, thread_count in schedule:
        if fail_reason is not None:
            break
        if time.perf_counter() >= deadline:
            fail_reason = "wall_time_cap"
            break
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(thread_count)
        env["JAX_PLATFORMS"] = "cpu"
        env["JAX_ENABLE_X64"] = "1"
        env.pop("SIMSOPT_BACKEND_MODE", None)
        started = time.perf_counter()
        with tempfile.NamedTemporaryFile(
            suffix=".json", prefix="nested_ls_banana_omp_", delete=False
        ) as handle:
            child_out = Path(handle.name)
        completed = subprocess.run(
            [python, str(child), str(child_out)],
            cwd=str(Path(__file__).resolve().parents[3]),
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        seconds = float(time.perf_counter() - started)
        if completed.returncode != 0 or not child_out.is_file():
            fail_reason = "banana_child_failed"
            rows.append(
                {
                    "repeat": int(repeat),
                    "omp_num_threads": int(thread_count),
                    "returncode": int(completed.returncode),
                    "seconds": seconds,
                    "stderr_tail": completed.stderr[-2000:],
                }
            )
            print(
                f"banana omp child fail threads={thread_count} rc={completed.returncode}",
                flush=True,
            )
            break
        payload = json.loads(child_out.read_text(encoding="utf-8"))
        child_out.unlink(missing_ok=True)
        pinned = bool(payload["omp_pinned"])
        any_unpinned = any_unpinned or (not pinned)
        row = {
            "repeat": int(repeat),
            "omp_num_threads": int(thread_count),
            "observed_omp_num_threads": payload["omp_num_threads"],
            "omp_pinned": pinned,
            "success": bool(payload["success"]),
            "bfgs_iter": int(payload["bfgs_iter"]),
            "newton_iter": int(payload["newton_iter"]),
            "bfgs_seconds": _json_finite(payload["bfgs_seconds"]),
            "newton_seconds": _json_finite(payload["newton_seconds"]),
            "seconds": _json_finite(payload["seconds"]),
            "iota": _json_finite(payload["iota"]),
            "G": _json_finite(payload["G"]),
            "coil_delta_inf": _json_finite(payload["coil_delta_inf"]),
        }
        rows.append(row)
        print(
            "banana omp"
            f" repeat={repeat} threads={thread_count}"
            f" s={payload['seconds']!r} bfgs_iter={payload['bfgs_iter']}"
            f" newton_iter={payload['newton_iter']} pinned={pinned}",
            flush=True,
        )
        if not pinned:
            fail_reason = "banana_omp_unpinned"
        elif not bool(payload["success"]):
            fail_reason = "banana_failed"
        elif float(payload["coil_delta_inf"]) != 0.0:
            fail_reason = "coil_moved"
    if fail_reason is None and not rows:
        fail_reason = "banana_child_failed"
    if fail_reason is None and any_unpinned:
        fail_reason = "banana_omp_unpinned"
    return NestedLsBananaOmpSweep(
        threads=tuple(int(value) for value in threads),
        repeats=int(repeats),
        rows=tuple(rows),
        any_unpinned=bool(any_unpinned),
        fail_closed_reason=fail_reason,
        runtime=runtime,
        provenance=provenance,
    )


@dataclass(frozen=True, slots=True)
class NestedLsChunkWarmProbe:
    """Warm in-process dense-assemble repeats at the LU endpoint.

    Cold first-touch is discarded per width. Not a production default
    switch and not a nested speed claim.
    """

    endpoint_sha256: str
    newton_stab: float
    warm_repeats: int
    rows: tuple[dict[str, object], ...]
    fail_closed_reason: str | None
    runtime: dict[str, object]
    provenance: dict[str, object]

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def evaluate_f3_b37_chunk_warm_probe(
    jax_boozer: BoozerSurfaceJAX,
    *,
    warm_repeats: int = F3_B37_CHUNK_WARM_REPEATS,
) -> NestedLsChunkWarmProbe:
    """Repeated warm Ĥ+stab I assemblies after one discarded warmup."""

    provenance = nested_ls_receipt_provenance()
    runtime = nested_ls_runtime_identity()
    loaded, coils, y_end, _iota, _g_const, _g_value, freeze_reason = (
        _freeze_dense_lu_walk_endpoint(jax_boozer)
    )
    sha = sha256_float64(loaded)

    def _empty(reason: str) -> NestedLsChunkWarmProbe:
        return NestedLsChunkWarmProbe(
            endpoint_sha256=sha,
            newton_stab=float(NESTED_LS_NEWTON_STAB),
            warm_repeats=int(warm_repeats),
            rows=(),
            fail_closed_reason=reason,
            runtime=runtime,
            provenance=provenance,
        )

    if freeze_reason is not None:
        return _empty(freeze_reason)
    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(jax_boozer)
    del _phi_hat
    operator = factor_reduced_nested_ls_schur(
        residual_fn,
        objective_fn,
        loaded,
        y_probe=np.asarray(y_end, dtype=np.float64),
    )
    rows: list[dict[str, object]] = []
    fail_reason: str | None = None
    for width in F3_B37_CHUNK_WIDTHS:
        jax.block_until_ready(
            materialize_stabilized_schur_dense(
                operator,
                float(NESTED_LS_NEWTON_STAB),
                chunk_batch_size=int(width),
            )
        )
        samples: list[float] = []
        for _repeat in range(int(warm_repeats)):
            started = time.perf_counter()
            jax.block_until_ready(
                materialize_stabilized_schur_dense(
                    operator,
                    float(NESTED_LS_NEWTON_STAB),
                    chunk_batch_size=int(width),
                )
            )
            samples.append(float(time.perf_counter() - started))
        mean = float(sum(samples) / len(samples))
        minimum = float(min(samples))
        print(
            f"chunk warm width={width} samples={samples} mean={mean:.3f}",
            flush=True,
        )
        rows.append(
            {
                "chunk_batch_size": int(width),
                "warm_repeats": int(warm_repeats),
                "dense_seconds": [_json_finite(value) for value in samples],
                "dense_seconds_mean": _json_finite(mean),
                "dense_seconds_min": _json_finite(minimum),
                "discarded_warmup": True,
            }
        )
    coil_after = float(
        np.linalg.norm(
            np.asarray(jax_boozer.biotsavart.x, dtype=np.float64) - coils,
            ord=np.inf,
        )
    )
    if fail_reason is None and coil_after != 0.0:
        fail_reason = "coil_moved"
    return NestedLsChunkWarmProbe(
        endpoint_sha256=sha,
        newton_stab=float(NESTED_LS_NEWTON_STAB),
        warm_repeats=int(warm_repeats),
        rows=tuple(rows),
        fail_closed_reason=fail_reason,
        runtime=runtime,
        provenance=provenance,
    )


def evaluate_f3_b37_nested_timing(
    native: BoozerSurface,
    jax_boozer: BoozerSurfaceJAX,
    *,
    b3_matched: bool,
    maxiter: int = NESTED_LS_NEWTON_MAXITER,
) -> NestedLsB37NestedTiming:
    """Time JAX reconstruct Schur walk vs native banana ``run_code``.

    Refuses unless B3 banana ``run_code`` already matched. Does not inherit
    F3 7.70×.
    """

    if not b3_matched:
        raise NestedLsB37TimingBlocked(
            "B37 nested timing only after B3 banana run_code physics match"
        )
    residual_fn, _objective_fn, _phi_hat = nested_ls_reduced_closures(jax_boozer)
    surface0 = np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64)
    native_start = np.asarray(native.surface.get_dofs(), dtype=np.float64)
    solution = solve_projected_y(residual_fn, surface0, np.zeros(2, dtype=np.float64))
    require_full_y_rank(solution)
    y_star = np.asarray(solution.solution, dtype=np.float64)
    banana_options = nested_ls_banana_run_code_options()
    original_options = replace_native_solver_options(
        native,
        {
            "newton_tol": banana_options["newton_tol"],
            "newton_maxiter": banana_options["newton_maxiter"],
            "bfgs_tol": banana_options["bfgs_tol"],
        },
    )
    native.surface.set_dofs(native_start)
    native_coils0 = np.asarray(native.biotsavart.x, dtype=np.float64)
    native.need_to_run_code = True
    banana_started = time.perf_counter()
    try:
        banana = native.run_code(float(y_star[0]), G=float(y_star[1]))
        banana_seconds = time.perf_counter() - banana_started
    finally:
        native.options = original_options
    if banana is None:
        raise RuntimeError("native banana run_code returned None.")
    native_coils1 = np.asarray(native.biotsavart.x, dtype=np.float64)
    jax_boozer.surface.set_dofs(surface0)
    walk_started = time.perf_counter()
    walk = run_reduced_nested_ls_schur_newton(
        jax_boozer,
        iota=float(y_star[0]),
        G=float(y_star[1]),
        maxiter=int(maxiter),
    )
    walk_seconds = time.perf_counter() - walk_started
    return NestedLsB37NestedTiming(
        b3_matched=True,
        comparable_operators=False,
        jax_walk_seconds=float(walk_seconds),
        jax_walk_iter=int(walk.iteration_count),
        jax_walk_iota=float(walk.iota),
        jax_walk_success=bool(walk.success),
        native_banana_seconds=float(banana_seconds),
        native_banana_iter=int(banana["iter"]),
        native_banana_iota=float(banana["iota"]),
        native_banana_success=bool(banana["success"]),
        coil_delta_inf=float(
            max(
                walk.coil_delta_inf,
                float(np.linalg.norm(native_coils1 - native_coils0, ord=np.inf)),
            )
        ),
        runtime=nested_ls_runtime_identity(),
    )


__all__ = [
    "ARCHIVED_START_QR_G",
    "ARCHIVED_START_QR_IOTA",
    "DEFAULT_F3_B37_GPU_LANE",
    "DEFAULT_F3_B37_NATIVE_LANE",
    "DEFAULT_FLAT675_BUNDLE_ROOT",
    "F3_B37_ADJOINT_COIL_SCAN",
    "F3_B37_ADJOINT_FD_EPSILON",
    "F3_B37_ADJOINT_LIVE_ETA_TOL",
    "F3_B37_ADJOINT_WALL_SECONDS",
    "F3_B37_BANANA_OMP_REPEATS",
    "F3_B37_BANANA_OMP_THREADS",
    "F3_B37_BANANA_OMP_WALL_SECONDS",
    "F3_B37_CAP512_WALK_EVIDENCE",
    "F3_B37_CHUNK_BANANA_WALL_SECONDS",
    "F3_B37_CHUNK_WARM_REPEATS",
    "F3_B37_CHUNK_WIDTHS",
    "F3_B37_DENSE_LU_ENDPOINT_G",
    "F3_B37_DENSE_LU_ENDPOINT_GRAD_L2",
    "F3_B37_DENSE_LU_ENDPOINT_IOTA",
    "F3_B37_DENSE_LU_ENDPOINT_SURFACE_SHA256",
    "F3_B37_DENSE_LU_WALK_EVIDENCE",
    "F3_B37_IFT_STAB",
    "F3_B37_STEP3_SURFACE_SHA256",
    "F3_B37_STEP4_CAP64_ETA_ACHIEVED",
    "F3_B37_STEP4_CAP64_RESIDUAL_L2",
    "F3_B37_STEP4_ETA_REQUESTED",
    "F3_B37_STEP5_SURFACE_SHA256",
    "F3_B37_STEP6_ARCH_FULL_RESTART",
    "F3_B37_STEP6_ARCH_HVP_BUDGET",
    "F3_B37_STEP6_ARCH_RESTARTS",
    "F3_B37_STEP6_ARCH_SOLVE_METHODS",
    "F3_B37_STEP6_ARCH_SWEEP_JAX_TOL",
    "F3_B37_STEP6_ARCH_WALL_SECONDS",
    "F3_B37_STEP6_CAP512_ETA_ACHIEVED",
    "F3_B37_STEP6_CAP_2048",
    "F3_B37_STEP6_CAP_2048_START_MAXITER",
    "F3_B37_STEP6_ETA_REQUESTED",
    "F3_B37_STEP6_G",
    "F3_B37_STEP6_GRAD_L2",
    "F3_B37_STEP6_IOTA",
    "F3_B37_STEP6_MATERIAL_RESIDUAL_RATIO",
    "F3_B37_STEP6_WALL_SECONDS_2048",
    "F3_B37_VOLUME_OUTER_WALL_SECONDS",
    "NestedLsB37NestedTiming",
    "NestedLsBananaOmpSweep",
    "NestedLsBoundedF3B37Result",
    "NestedLsChunkBananaProbe",
    "NestedLsChunkWarmProbe",
    "NestedLsCountedMatvec",
    "NestedLsEndpointAdjointProbe",
    "NestedLsFlatNativeB37Probe",
    "NestedLsSchurNewtonStepProbe",
    "NestedLsSchurNewtonWalkProbe",
    "NestedLsStep2ForcingProbe",
    "NestedLsStep4ForcingProbe",
    "NestedLsStep6ArchitectureProbe",
    "NestedLsStep6ForcingProbe",
    "NestedLsVolumeOuterProbe",
    "archived_f3_b37_lanes_available",
    "archived_flat675_bundle_available",
    "dump_strict_json",
    "evaluate_f3_b37_banana_omp_sweep",
    "evaluate_f3_b37_bounded_probe",
    "evaluate_f3_b37_chunk_banana_probe",
    "evaluate_f3_b37_chunk_warm_probe",
    "evaluate_f3_b37_endpoint_adjoint_probe",
    "evaluate_f3_b37_flat_native_probe",
    "evaluate_f3_b37_nested_timing",
    "evaluate_f3_b37_schur_newton_step",
    "evaluate_f3_b37_schur_newton_walk",
    "evaluate_f3_b37_step2_forcing_probe",
    "evaluate_f3_b37_step4_forcing_probe",
    "evaluate_f3_b37_step6_architecture_probe",
    "evaluate_f3_b37_step6_forcing_probe",
    "evaluate_f3_b37_volume_outer_probe",
    "float64_ulps",
    "gmres_doubling_cycle_budget",
    "kib_to_gib",
    "last_step_meets_forcing",
    "load_archived_nested_ls_pair",
    "load_flat675_lane_blocks",
    "nested_ls_omp_threads_pinned",
    "nested_ls_receipt_provenance",
    "nested_ls_runtime_identity",
    "nested_ls_threading_env",
    "predict_start_at_cap_wall_seconds",
    "replace_native_solver_options",
    "sha256_file",
    "sha256_float64",
    "unpreconditioned_gmres_is_insufficient",
    "write_strict_json",
]
