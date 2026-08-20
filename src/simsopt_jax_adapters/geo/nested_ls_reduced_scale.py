"""Build the archived 255×64 nested-LS problem from the F3 frozen bundle.

Gate 1 of the reduced nested-LS track. Coils come from the bundle's native
Biot-Savart JSON; the surface comes from the flat-675 runtime spec. This
does not evaluate F3's fused objective and does not inherit the 7.70× claim.
"""

from __future__ import annotations

import json
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
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
    nested_ls_physics_newton_kwargs,
)
from simsopt_jax_adapters.geo.nested_ls_reduced import (
    nested_ls_reduced_closures,
    pack_surface_and_y,
    reduced_penalty_gradient,
    reduced_penalty_hvp,
    require_full_y_rank,
    run_reduced_nested_ls_newton,
    solve_projected_y,
)

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
    if coils.shape != native_field.x.shape:
        raise ValueError(
            "coil coordinates shape "
            f"{coils.shape} does not match archived BiotSavart.x "
            f"{native_field.x.shape}."
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


def _peak_rss_kib() -> int:
    """Linux ``ru_maxrss`` is kibibytes and is the process peak, not a delta."""

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
    native_rejudge_iota: float
    native_rejudge_g: float
    native_rejudge_coil_delta_inf: float
    native_rejudge_seconds: float
    full_walk_attempted: bool

    def as_payload(self) -> dict[str, object]:
        return asdict(self)


def evaluate_f3_b37_bounded_probe(
    native: BoozerSurface,
    jax_boozer: BoozerSurfaceJAX,
    *,
    one_newton_step: bool = False,
) -> NestedLsBoundedF3B37Result:
    """C++ full reference plus JAX ``y*`` / ``g`` / one HVP.

    The JAX lane starts from the original F3 surface, not from the C++
    endpoint. One Newton step is opt-in: autodiff-through-QR GMRES is a
    many-HVP solve and is not the default bounded gate. Does not launch
    a ten-step reduced walk.
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
        rejudge_iota = float("nan")
        rejudge_g = float("nan")
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
    )


__all__ = [
    "ARCHIVED_START_QR_G",
    "ARCHIVED_START_QR_IOTA",
    "DEFAULT_F3_B37_GPU_LANE",
    "DEFAULT_F3_B37_NATIVE_LANE",
    "DEFAULT_FLAT675_BUNDLE_ROOT",
    "NestedLsBoundedF3B37Result",
    "archived_f3_b37_lanes_available",
    "archived_flat675_bundle_available",
    "evaluate_f3_b37_bounded_probe",
    "load_archived_nested_ls_pair",
    "load_flat675_lane_blocks",
]
