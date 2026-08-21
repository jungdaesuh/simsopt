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
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np
import scipy
import simsopt
import simsoptpp
from numpy.typing import NDArray
from simsopt._core.json import GSONDecoder
from simsopt.field import BiotSavart
from simsopt.geo import BoozerSurface, SurfaceXYZTensorFourier, Volume

from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.flat675.bundle import load_flat675_bundle
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_CONSTRAINT_WEIGHT,
    NESTED_LS_NEWTON_MAXITER,
    NESTED_LS_NEWTON_TOL,
    NESTED_LS_WEIGHT_INV_MODB,
    nested_ls_banana_run_code_options,
    nested_ls_physics_newton_kwargs,
)
from simsopt_jax_adapters.geo.nested_ls_reduced import (
    NestedLsB37TimingBlocked,
    compare_ad_qr_and_schur_hvp,
    factor_reduced_nested_ls_schur,
    nested_ls_reduced_closures,
    pack_surface_and_y,
    reduced_penalty_gradient,
    reduced_penalty_gradient_envelope,
    reduced_penalty_hvp,
    require_full_y_rank,
    run_reduced_nested_ls_newton,
    run_reduced_nested_ls_schur_newton,
    solve_projected_y,
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


def nested_ls_runtime_identity() -> dict[str, object]:
    """Host and JAX device identity for an evidence document."""

    return {
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS"),
        "jax_enable_x64_env": os.environ.get("JAX_ENABLE_X64"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def dump_strict_json(payload: dict[str, object]) -> str:
    """Serialize evidence with ``allow_nan=False`` so NaN cannot sneak in."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


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
    steps: tuple[dict[str, object], ...]
    jax_iota: float
    jax_g: float
    jax_surface_sha256: str
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
) -> NestedLsSchurNewtonWalkProbe:
    """Frozen-coil Schur Newton walk from F3 B37, then C++ reconstruct rejudge.

    Uses GMRES, not per-step dense LU. Coils stay frozen.
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
        steps=tuple(asdict(record) for record in walk.steps),
        jax_iota=float(walk.iota),
        jax_g=float(walk.G),
        jax_surface_sha256=sha256_float64(jax_surface),
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


def replace_native_solver_options(
    native: BoozerSurface,
    overlay: dict[str, object],
) -> dict[str, object]:
    """Replace ``native.options`` with a copy plus ``overlay``. Return the old dict."""

    original = native.options
    native.options = {**original, **overlay}
    return original


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
    "NestedLsB37NestedTiming",
    "NestedLsBoundedF3B37Result",
    "NestedLsFlatNativeB37Probe",
    "NestedLsSchurNewtonStepProbe",
    "NestedLsSchurNewtonWalkProbe",
    "archived_f3_b37_lanes_available",
    "archived_flat675_bundle_available",
    "dump_strict_json",
    "evaluate_f3_b37_bounded_probe",
    "evaluate_f3_b37_flat_native_probe",
    "evaluate_f3_b37_nested_timing",
    "evaluate_f3_b37_schur_newton_step",
    "evaluate_f3_b37_schur_newton_walk",
    "float64_ulps",
    "kib_to_gib",
    "load_archived_nested_ls_pair",
    "load_flat675_lane_blocks",
    "nested_ls_receipt_provenance",
    "nested_ls_runtime_identity",
    "replace_native_solver_options",
    "sha256_file",
    "sha256_float64",
    "write_strict_json",
]
