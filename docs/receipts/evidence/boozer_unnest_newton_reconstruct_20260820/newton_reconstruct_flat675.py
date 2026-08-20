#!/usr/bin/env python
"""Analysis-only Newton reconstruct: C++ BoozerSurface as judge vs un-nest s.

Freeze coils, start from a flat-675 surface, call native C++ Boozer LS Newton
(and exact Newton on the collocation grid). Not a product test. Does not
modify any repository.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "16")

REPO = Path("/home/jungdaesuh/code/columbia/simsopt-genuine675-fairbar")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarks.validation_ladder_common import (  # noqa: E402
    apply_requested_platform,
    bootstrap_local_simsopt,
    require_requested_platform_runtime,
    require_x64_runtime,
)

apply_requested_platform("cpu")
import jax  # noqa: E402

require_x64_runtime(jax, context="Newton reconstruct analysis")
require_requested_platform_runtime(
    jax,
    requested_platform="cpu",
    context="Newton reconstruct analysis",
)
bootstrap_local_simsopt()

import numpy as np  # noqa: E402
from simsopt.geo import BoozerSurface, SurfaceXYZTensorFourier, Volume  # noqa: E402

from benchmarks.fixed_state_genuine_675_input_manifest import (  # noqa: E402
    validate_frozen_genuine_675_input_bundle,
)
from simsopt_jax.runtime.single_stage_fullspace_675 import (  # noqa: E402
    Fullspace675Candidate,
)
from simsopt_jax_adapters.geo.single_stage_fullspace_675 import (  # noqa: E402
    Fullspace675NativeBoozerMaterial,
    Fullspace675NativeBoozerSystemMaterializer,
)

BUNDLE_MANIFEST = (
    Path.home()
    / "simsopt_mixed_artifacts"
    / "genuine675-r3-input-1c23f6c5-20260721-r1"
    / "manifest.json"
)
F3_B37_LANE = (
    Path.home()
    / "simsopt_mixed_artifacts"
    / "flat675_fused_campaign"
    / "20260819T163816Z-pairs-b37-2751085"
    / "pair2-l1"
    / "lane.json"
)
OUT_JSON = Path("/tmp/newton_reconstruct_flat675.json")
TARGET_LABEL = 0.1
LS_CONSTRAINT_WEIGHT = 1.0
WEIGHT_INV_MODB = True
LS_NEWTON_MAXITER = 10
LS_NEWTON_TOL = 1e-13
LS_NEWTON_STAB = 1.0e-4
EXACT_NEWTON_MAXITER = 20
EXACT_NEWTON_TOL = 1e-13


def _log(msg: str) -> None:
    print(msg, flush=True)


def _finite_stats(name: str, array: np.ndarray) -> dict[str, object]:
    values = np.asarray(array, dtype=np.float64)
    return {
        name + "_shape": list(values.shape),
        name + "_l2": float(np.linalg.norm(values)),
        name + "_inf": float(np.linalg.norm(values, ord=np.inf)),
        name + "_finite": bool(np.all(np.isfinite(values))),
    }


def _qr_inner(design_matrix: np.ndarray, rhs: np.ndarray) -> dict[str, object]:
    matrix = np.asarray(design_matrix, dtype=np.float64)
    vector = np.asarray(rhs, dtype=np.float64)
    orthogonal, triangular = np.linalg.qr(matrix, mode="reduced")
    solution = np.linalg.solve(triangular, orthogonal.T @ vector)
    residual = matrix @ solution - vector
    singular_values = np.linalg.svd(triangular, compute_uv=False)
    return {
        "iota": float(solution[0]),
        "G": float(solution[1]),
        "residual_l2": float(np.linalg.norm(residual)),
        "relative_fit_residual": float(
            np.linalg.norm(residual) / max(np.linalg.norm(vector), np.finfo(np.float64).tiny)
        ),
        "singular_values": [float(v) for v in singular_values],
        "matrix_shape": list(matrix.shape),
    }


def _collocation_quadpoints(nfp: int, mpol: int, ntor: int) -> tuple[np.ndarray, np.ndarray]:
    phis = np.linspace(0.0, 1.0 / nfp, 2 * ntor + 1, endpoint=False)
    thetas = np.linspace(0.0, 1.0, 2 * mpol + 1, endpoint=False)
    return phis, thetas


def _clamped_dims(surface: SurfaceXYZTensorFourier) -> list[bool]:
    values = getattr(surface, "clamped_dims", [False, False, False])
    return [bool(value) for value in values]


def _copy_surface(surface: SurfaceXYZTensorFourier) -> SurfaceXYZTensorFourier:
    copied = SurfaceXYZTensorFourier(
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        clamped_dims=_clamped_dims(surface),
        quadpoints_phi=np.asarray(surface.quadpoints_phi, dtype=np.float64),
        quadpoints_theta=np.asarray(surface.quadpoints_theta, dtype=np.float64),
    )
    copied.set_dofs(np.asarray(surface.get_dofs(), dtype=np.float64))
    return copied


def _collocation_surface(surface: SurfaceXYZTensorFourier) -> SurfaceXYZTensorFourier:
    phis, thetas = _collocation_quadpoints(surface.nfp, surface.mpol, surface.ntor)
    copied = SurfaceXYZTensorFourier(
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        clamped_dims=_clamped_dims(surface),
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    copied.set_dofs(np.asarray(surface.get_dofs(), dtype=np.float64))
    return copied


def _ls_state_metrics(
    booz: BoozerSurface,
    iota: float,
    G: float,
) -> dict[str, object]:
    surface = booz.surface
    x = np.concatenate(
        (np.asarray(surface.get_dofs(), dtype=np.float64), [float(iota), float(G)])
    )
    t0 = time.perf_counter()
    val, dval = booz.boozer_penalty_constraints_vectorized(
        x,
        derivatives=1,
        constraint_weight=LS_CONSTRAINT_WEIGHT,
        optimize_G=True,
        weight_inv_modB=WEIGHT_INV_MODB,
    )
    elapsed = time.perf_counter() - t0
    volume = float(Volume(surface).J())
    metrics = {
        "ls_objective": float(val),
        "ls_grad_seconds": elapsed,
        "volume": volume,
        "volume_minus_target": volume - TARGET_LABEL,
        "iota": float(iota),
        "G": float(G),
        "surface_dof_count": int(np.asarray(surface.get_dofs()).size),
        "nphi": int(np.asarray(surface.quadpoints_phi).size),
        "ntheta": int(np.asarray(surface.quadpoints_theta).size),
        "stellsym": bool(surface.stellsym),
        "mpol": int(surface.mpol),
        "ntor": int(surface.ntor),
        "nfp": int(surface.nfp),
    }
    metrics.update(_finite_stats("ls_grad", np.asarray(dval, dtype=np.float64)))
    return metrics


def _run_ls_newton(
    biot_savart,
    surface: SurfaceXYZTensorFourier,
    iota: float,
    G: float,
) -> dict[str, object]:
    working = _copy_surface(surface)
    s0 = np.asarray(working.get_dofs(), dtype=np.float64).copy()
    coil_x0 = np.asarray(biot_savart.x, dtype=np.float64).copy()
    booz = BoozerSurface(
        biot_savart,
        working,
        Volume(working),
        TARGET_LABEL,
        constraint_weight=LS_CONSTRAINT_WEIGHT,
        options={
            "verbose": True,
            "newton_tol": LS_NEWTON_TOL,
            "newton_maxiter": LS_NEWTON_MAXITER,
            "weight_inv_modB": WEIGHT_INV_MODB,
            "limited_memory": False,
        },
    )
    before = _ls_state_metrics(booz, iota, G)
    t0 = time.perf_counter()
    booz.need_to_run_code = True
    result = booz.minimize_boozer_penalty_constraints_newton(
        constraint_weight=LS_CONSTRAINT_WEIGHT,
        iota=float(iota),
        G=float(G),
        verbose=True,
        tol=LS_NEWTON_TOL,
        maxiter=LS_NEWTON_MAXITER,
        stab=LS_NEWTON_STAB,
        weight_inv_modB=WEIGHT_INV_MODB,
    )
    elapsed = time.perf_counter() - t0
    s1 = np.asarray(working.get_dofs(), dtype=np.float64)
    after = _ls_state_metrics(booz, float(result["iota"]), float(result["G"]))
    coil_x1 = np.asarray(biot_savart.x, dtype=np.float64)
    return {
        "kind": "ls_newton_polish",
        "constraint_weight": LS_CONSTRAINT_WEIGHT,
        "stab": LS_NEWTON_STAB,
        "maxiter_cap": LS_NEWTON_MAXITER,
        "tol": LS_NEWTON_TOL,
        "seconds": elapsed,
        "success": bool(result.get("success")),
        "iter": int(result.get("iter", -1)),
        "before": before,
        "after": after,
        "delta_surface_l2": float(np.linalg.norm(s1 - s0)),
        "delta_surface_inf": float(np.linalg.norm(s1 - s0, ord=np.inf)),
        "delta_iota": float(result["iota"] - iota),
        "delta_G": float(result["G"] - G),
        "coil_x_delta_inf": float(np.linalg.norm(coil_x1 - coil_x0, ord=np.inf)),
        "result_keys": sorted(result.keys()),
    }


def _run_exact_newton(
    biot_savart,
    surface: SurfaceXYZTensorFourier,
    iota: float,
    G: float,
    *,
    collocation: bool,
) -> dict[str, object]:
    working = _collocation_surface(surface) if collocation else _copy_surface(surface)
    s0 = np.asarray(working.get_dofs(), dtype=np.float64).copy()
    coil_x0 = np.asarray(biot_savart.x, dtype=np.float64).copy()
    grid = {
        "nphi": int(np.asarray(working.quadpoints_phi).size),
        "ntheta": int(np.asarray(working.quadpoints_theta).size),
        "stellsym": bool(working.stellsym),
        "collocation": collocation,
    }
    try:
        mask = working.get_stellsym_mask()
        grid["stellsym_mask_true_count"] = int(np.count_nonzero(mask))
        grid["stellsym_mask_ok"] = True
    except Exception as exc:  # noqa: BLE001 — analysis of the exact-Newton grid gate
        grid["stellsym_mask_ok"] = False
        grid["stellsym_mask_error"] = f"{type(exc).__name__}: {exc}"
        if not collocation:
            return {
                "kind": "exact_newton",
                "constraint_weight": None,
                "grid": grid,
                "skipped": True,
                "skip_reason": "stellsym collocation gate failed on the 675 residual grid",
            }
        raise

    booz = BoozerSurface(
        biot_savart,
        working,
        Volume(working),
        TARGET_LABEL,
        constraint_weight=None,
        options={
            "verbose": True,
            "newton_tol": EXACT_NEWTON_TOL,
            "newton_maxiter": EXACT_NEWTON_MAXITER,
        },
    )
    volume_before = float(Volume(working).J())
    t0 = time.perf_counter()
    booz.need_to_run_code = True
    result = booz.solve_residual_equation_exactly_newton(
        iota=float(iota),
        G=float(G),
        tol=EXACT_NEWTON_TOL,
        maxiter=EXACT_NEWTON_MAXITER,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    s1 = np.asarray(working.get_dofs(), dtype=np.float64)
    residual = np.asarray(result["residual"], dtype=np.float64)
    volume_after = float(Volume(working).J())
    coil_x1 = np.asarray(biot_savart.x, dtype=np.float64)
    payload = {
        "kind": "exact_newton",
        "constraint_weight": None,
        "maxiter_cap": EXACT_NEWTON_MAXITER,
        "tol": EXACT_NEWTON_TOL,
        "seconds": elapsed,
        "success": bool(result.get("success")),
        "iter": int(result.get("iter", -1)),
        "grid": grid,
        "volume_before": volume_before,
        "volume_after": volume_after,
        "volume_minus_target_before": volume_before - TARGET_LABEL,
        "volume_minus_target_after": volume_after - TARGET_LABEL,
        "iota_before": float(iota),
        "G_before": float(G),
        "iota_after": float(result["iota"]),
        "G_after": None if result.get("G") is None else float(result["G"]),
        "delta_surface_l2": float(np.linalg.norm(s1 - s0)),
        "delta_surface_inf": float(np.linalg.norm(s1 - s0, ord=np.inf)),
        "delta_iota": float(result["iota"] - iota),
        "delta_G": (
            None if result.get("G") is None else float(result["G"] - G)
        ),
        "coil_x_delta_inf": float(np.linalg.norm(coil_x1 - coil_x0, ord=np.inf)),
    }
    payload.update(_finite_stats("exact_residual", residual))
    return payload


def _candidate_from_lane(path: Path) -> tuple[Fullspace675Candidate, dict[str, object]]:
    lane = json.loads(path.read_bytes())
    block = lane["result"]["endpoint_candidate"]
    candidate = Fullspace675Candidate(
        coil_coordinates=tuple(float(v) for v in block["coil_coordinates"]),
        vessel_coordinates=tuple(float(v) for v in block["vessel_coordinates"]),
        surface_coordinates=tuple(float(v) for v in block["surface_coordinates"]),
    )
    meta = {
        "lane": lane.get("lane"),
        "process_wall_seconds": lane.get("process_wall_seconds"),
        "endpoint_inner_state": lane["result"].get("endpoint_inner_state"),
        "endpoint_objective": lane["result"].get("objective_value"),
        "nfev": lane["result"].get("nfev"),
        "nit": lane["result"].get("nit"),
        "path": str(path),
    }
    return candidate, meta


def _reconstruct_one(
    name: str,
    candidate: Fullspace675Candidate,
    materializer: Fullspace675NativeBoozerSystemMaterializer,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    _log(f"\n=== materialize {name} ===")
    t0 = time.perf_counter()
    physical = materializer.materialize_physical_state(candidate)
    materialize_seconds = time.perf_counter() - t0
    qr = _qr_inner(physical.system.design_matrix, physical.system.right_hand_side)
    _log(
        f"{name}: QR iota={qr['iota']:.16f} G={qr['G']:.16f} "
        f"resid_l2={qr['residual_l2']:.6e} rel={qr['relative_fit_residual']:.6e} "
        f"A={qr['matrix_shape']} materialize={materialize_seconds:.2f}s"
    )
    surface = physical.surface
    biot_savart = physical.biot_savart
    report: dict[str, object] = {
        "name": name,
        "materialize_seconds": materialize_seconds,
        "qr_inner": qr,
        "surface_dof_count": int(np.asarray(surface.get_dofs()).size),
        "nphi": int(np.asarray(surface.quadpoints_phi).size),
        "ntheta": int(np.asarray(surface.quadpoints_theta).size),
        "stellsym": bool(surface.stellsym),
        "mpol": int(surface.mpol),
        "ntor": int(surface.ntor),
        "nfp": int(surface.nfp),
        "volume_on_residual_grid": float(Volume(surface).J()),
        "coil_dof_count": int(np.asarray(biot_savart.x).size),
    }
    if extra:
        report["source"] = extra

    _log(f"=== LS Newton polish {name} ===")
    try:
        report["ls_newton"] = _run_ls_newton(
            biot_savart, surface, qr["iota"], qr["G"]
        )
        ls = report["ls_newton"]
        _log(
            f"{name} LS: success={ls['success']} iter={ls['iter']} "
            f"||ds||_2={ls['delta_surface_l2']:.6e} ||ds||_inf={ls['delta_surface_inf']:.6e} "
            f"d_iota={ls['delta_iota']:.6e} d_G={ls['delta_G']:.6e} "
            f"grad_inf {ls['before']['ls_grad_inf']:.3e} -> {ls['after']['ls_grad_inf']:.3e} "
            f"in {ls['seconds']:.2f}s coils_frozen={ls['coil_x_delta_inf']==0.0}"
        )
    except Exception as exc:  # noqa: BLE001
        report["ls_newton"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _log(f"{name} LS FAILED: {exc}")

    _log(f"=== exact Newton on 675 residual grid {name} ===")
    try:
        report["exact_newton_residual_grid"] = _run_exact_newton(
            biot_savart, surface, qr["iota"], qr["G"], collocation=False
        )
        _log(f"{name} exact-on-675-grid: {report['exact_newton_residual_grid']}")
    except Exception as exc:  # noqa: BLE001
        report["exact_newton_residual_grid"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _log(f"{name} exact-on-675-grid FAILED: {exc}")

    _log(f"=== exact Newton on collocation grid {name} ===")
    try:
        report["exact_newton_collocation"] = _run_exact_newton(
            biot_savart, surface, qr["iota"], qr["G"], collocation=True
        )
        ex = report["exact_newton_collocation"]
        _log(
            f"{name} exact-collocation: success={ex.get('success')} iter={ex.get('iter')} "
            f"||ds||_2={ex.get('delta_surface_l2')} resid_inf={ex.get('exact_residual_inf')} "
            f"in {ex.get('seconds')}s"
        )
    except Exception as exc:  # noqa: BLE001
        report["exact_newton_collocation"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _log(f"{name} exact-collocation FAILED: {exc}")
    return report


def main() -> int:
    _log(f"repo={REPO}")
    _log(f"python={sys.executable}")
    _log(f"jax backend={jax.default_backend()} devices={jax.devices()}")
    _log("validating frozen genuine-675 bundle...")
    t0 = time.perf_counter()
    validated = validate_frozen_genuine_675_input_bundle(
        BUNDLE_MANIFEST,
        source_repo=REPO,
    )
    _log(f"bundle validated in {time.perf_counter() - t0:.2f}s")
    manifest = validated.manifest
    material = Fullspace675NativeBoozerMaterial.from_runtime_spec(
        validated.runtime_spec,
        native_biot_savart_payload=validated.native_biot_savart_payload,
    )
    materializer = Fullspace675NativeBoozerSystemMaterializer(
        material=material,
        policy=manifest.boozer_construction_policy,
    )
    start = manifest.candidate
    f3_candidate, f3_meta = _candidate_from_lane(F3_B37_LANE)

    payload: dict[str, object] = {
        "analysis": "newton_reconstruct_flat675",
        "repo": str(REPO),
        "python": sys.executable,
        "bundle": str(BUNDLE_MANIFEST),
        "requirements_named_in_code": {
            "same_B_operator": "native BiotSavart payload + candidate coil dofs",
            "un_nest_inner": "QR on the 48960x2 design matrix, surface frozen",
            "nested_ls_judge": (
                "C++ BoozerSurface.minimize_boozer_penalty_constraints_newton "
                "constraint_weight=1.0, G free, weight_inv_modB=True, stab=1e-4"
            ),
            "nested_exact_judge": (
                "C++ solve_residual_equation_exactly_newton on the stellsym "
                "collocation grid, constraint_weight=None, volume=0.1 exact"
            ),
            "label": "Volume target 0.1; LS penalizes, exact enforces",
            "grid": "675 residual grid is 255x64, not the exact collocation grid",
            "coils_frozen": "biot_savart.x must not change",
            "GATE_1_is_not_this_check": True,
        },
        "points": [],
    }
    payload["points"].append(
        _reconstruct_one("archived_start", start, materializer)
    )
    payload["points"].append(
        _reconstruct_one("f3_b37_pair2_l1_endpoint", f3_candidate, materializer, f3_meta)
    )
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _log(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
