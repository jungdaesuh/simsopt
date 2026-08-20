#!/usr/bin/env python
"""Try both Boozer purposes without nested-in-outer: LS, exact, LS→exact handoff.

Analysis only. Coils frozen. C++ residual + C++ LS Newton as judges.
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

require_x64_runtime(jax, context="LS/exact purpose try")
require_requested_platform_runtime(
    jax,
    requested_platform="cpu",
    context="LS/exact purpose try",
)
bootstrap_local_simsopt()

import numpy as np  # noqa: E402
from simsopt.geo import BoozerSurface, SurfaceXYZTensorFourier, Volume  # noqa: E402
from simsopt.geo.boozersurface import (  # noqa: E402
    _boozer_iterate_is_persistable,
    boozer_surface_residual,
)

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
OUT_JSON = Path("/tmp/boozer_ls_exact_purpose.json")
TARGET_LABEL = 0.1
LS_CONSTRAINT_WEIGHT = 1.0
WEIGHT_INV_MODB = True
LS_NEWTON_MAXITER = 10
LS_NEWTON_TOL = 1e-13
LS_NEWTON_STAB = 1.0e-4
EXACT_NEWTON_MAXITER = 40
EXACT_NEWTON_TOL = 1e-13
LS_MATCH_IOTA = 1.0e-8
LS_MATCH_S_INF = 1.0e-8
LS_PURPOSE_IOTA = 1.0e-3
LS_PURPOSE_S_INF = 1.0e-3
EXACT_PURPOSE_RESID = 1.0e-6


def _log(msg: str) -> None:
    print(msg, flush=True)


def _clamped_dims(surface: SurfaceXYZTensorFourier) -> list[bool]:
    values = getattr(surface, "clamped_dims", [False, False, False])
    return [bool(value) for value in values]


def _copy_surface(
    surface: SurfaceXYZTensorFourier,
    *,
    collocation: bool = False,
) -> SurfaceXYZTensorFourier:
    if collocation:
        phis = np.linspace(0.0, 1.0 / surface.nfp, 2 * surface.ntor + 1, endpoint=False)
        thetas = np.linspace(0.0, 1.0, 2 * surface.mpol + 1, endpoint=False)
    else:
        phis = np.asarray(surface.quadpoints_phi, dtype=np.float64)
        thetas = np.asarray(surface.quadpoints_theta, dtype=np.float64)
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


def _qr_inner(design_matrix: np.ndarray, rhs: np.ndarray) -> dict[str, object]:
    matrix = np.asarray(design_matrix, dtype=np.float64)
    vector = np.asarray(rhs, dtype=np.float64)
    orthogonal, triangular = np.linalg.qr(matrix, mode="reduced")
    solution = np.linalg.solve(triangular, orthogonal.T @ vector)
    residual = matrix @ solution - vector
    return {
        "iota": float(solution[0]),
        "G": float(solution[1]),
        "residual_l2": float(np.linalg.norm(residual)),
        "relative_fit_residual": float(
            np.linalg.norm(residual)
            / max(np.linalg.norm(vector), np.finfo(np.float64).tiny)
        ),
        "matrix_shape": list(matrix.shape),
    }


def _exact_masked_norm(booz: BoozerSurface, iota: float, G: float) -> dict[str, float]:
    surface = booz.surface
    mask = np.concatenate(
        [surface.get_stellsym_mask()[..., None]] * 3,
        axis=2,
    )
    if surface.stellsym:
        mask[0, 0, 0] = False
    residual_out = boozer_surface_residual(
        surface, iota, G, booz.biotsavart, derivatives=0
    )
    residual = np.asarray(residual_out[0], dtype=np.float64).reshape(-1)
    label_gap = float(booz.label.J() - booz.targetlabel)
    masked = np.concatenate((residual[mask.flatten()], [label_gap]))
    return {
        "residual_inf": float(np.linalg.norm(residual, ord=np.inf)),
        "residual_l2": float(np.linalg.norm(residual)),
        "masked_l2": float(np.linalg.norm(masked)),
        "label_gap": label_gap,
        "volume": float(Volume(surface).J()),
    }


def _run_ls_newton(biot_savart, surface, iota: float, G: float) -> dict[str, object]:
    working = _copy_surface(surface)
    s0 = np.asarray(working.get_dofs(), dtype=np.float64).copy()
    coil0 = np.asarray(biot_savart.x, dtype=np.float64).copy()
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
    x = np.concatenate((s0, [float(iota), float(G)]))
    _, grad0 = booz.boozer_penalty_constraints_vectorized(
        x,
        derivatives=1,
        constraint_weight=LS_CONSTRAINT_WEIGHT,
        optimize_G=True,
        weight_inv_modB=WEIGHT_INV_MODB,
    )
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
    x1 = np.concatenate((s1, [float(result["iota"]), float(result["G"])]))
    _, grad1 = booz.boozer_penalty_constraints_vectorized(
        x1,
        derivatives=1,
        constraint_weight=LS_CONSTRAINT_WEIGHT,
        optimize_G=True,
        weight_inv_modB=WEIGHT_INV_MODB,
    )
    d_iota = float(result["iota"] - iota)
    d_s_inf = float(np.linalg.norm(s1 - s0, ord=np.inf))
    d_s_l2 = float(np.linalg.norm(s1 - s0))
    grad0_inf = float(np.linalg.norm(np.asarray(grad0), ord=np.inf))
    grad1_inf = float(np.linalg.norm(np.asarray(grad1), ord=np.inf))
    match = bool(
        result.get("success")
        and abs(d_iota) < LS_MATCH_IOTA
        and d_s_inf < LS_MATCH_S_INF
        and grad0_inf < 1.0e-12
    )
    purpose = bool(abs(d_iota) < LS_PURPOSE_IOTA and d_s_inf < LS_PURPOSE_S_INF)
    return {
        "mode": "ls",
        "success": bool(result.get("success")),
        "iter": int(result.get("iter", -1)),
        "seconds": elapsed,
        "iota_before": float(iota),
        "iota_after": float(result["iota"]),
        "G_before": float(G),
        "G_after": float(result["G"]),
        "delta_iota": d_iota,
        "delta_G": float(result["G"] - G),
        "delta_surface_l2": d_s_l2,
        "delta_surface_inf": d_s_inf,
        "grad_inf_before": grad0_inf,
        "grad_inf_after": grad1_inf,
        "volume_before": float(Volume(_copy_surface(surface)).J()),
        "volume_after": float(Volume(working).J()),
        "coil_x_delta_inf": float(
            np.linalg.norm(np.asarray(biot_savart.x) - coil0, ord=np.inf)
        ),
        "match_nested_ls": match,
        "purpose_close": purpose,
        "surface_dofs": s1.tolist() if not match else None,
        "iota": float(result["iota"]),
        "G": float(result["G"]),
        "_working": working,
    }


def _run_exact_newton(
    biot_savart,
    surface,
    iota: float,
    G: float,
    *,
    seed: str,
) -> dict[str, object]:
    working = _copy_surface(surface, collocation=True)
    s0 = np.asarray(working.get_dofs(), dtype=np.float64).copy()
    coil0 = np.asarray(biot_savart.x, dtype=np.float64).copy()
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
    before = _exact_masked_norm(booz, iota, G)
    mask = np.concatenate(
        [working.get_stellsym_mask()[..., None]] * 3,
        axis=2,
    )
    if working.stellsym:
        mask[0, 0, 0] = False
    mask = mask.flatten()
    history: list[dict[str, float]] = []
    x = np.concatenate((s0, [float(iota), float(G)]))
    current_iota = float(iota)
    current_G = float(G)
    t0 = time.perf_counter()
    residual, jacobian = boozer_surface_residual(
        working, current_iota, current_G, biot_savart, derivatives=1
    )
    initial_norm = None
    iteration = 0
    norm = 1.0e6
    while iteration < EXACT_NEWTON_MAXITER:
        rhs = np.concatenate(
            (
                np.asarray(residual, dtype=np.float64)[mask],
                [float(booz.label.J() - TARGET_LABEL)],
            )
        )
        norm = float(np.linalg.norm(rhs))
        if initial_norm is None:
            initial_norm = norm
        history.append(
            {
                "iter": float(iteration),
                "masked_l2": norm,
                "iota": current_iota,
                "G": current_G,
                "volume": float(Volume(working).J()),
            }
        )
        if norm <= EXACT_NEWTON_TOL:
            break
        system = np.vstack(
            (
                np.asarray(jacobian, dtype=np.float64)[mask, :],
                np.concatenate((booz._label_surface_gradient(), [0.0, 0.0])),
            )
        )
        step = np.linalg.solve(system, rhs)
        step = step + np.linalg.solve(system, rhs - system @ step)
        x = x - step
        working.set_dofs(x[:-2])
        current_iota = float(x[-2])
        current_G = float(x[-1])
        iteration += 1
        residual, jacobian = boozer_surface_residual(
            working, current_iota, current_G, biot_savart, derivatives=1
        )
    elapsed = time.perf_counter() - t0
    success = bool(norm <= EXACT_NEWTON_TOL)
    persist = bool(
        _boozer_iterate_is_persistable(success, norm, initial_norm)
    )
    if not persist:
        working.set_dofs(s0)
        current_iota = float(iota)
        current_G = float(G)
    after = _exact_masked_norm(booz, current_iota, current_G)
    s1 = np.asarray(working.get_dofs(), dtype=np.float64)
    return {
        "mode": "exact",
        "seed": seed,
        "success": success,
        "persisted": persist,
        "iter": iteration,
        "seconds": elapsed,
        "history_head": history[:5],
        "history_tail": history[-5:],
        "history_len": len(history),
        "min_masked_l2": min(entry["masked_l2"] for entry in history),
        "iota_before": float(iota),
        "iota_after": current_iota,
        "G_before": float(G),
        "G_after": current_G,
        "delta_iota": current_iota - float(iota),
        "delta_G": current_G - float(G),
        "delta_surface_l2": float(np.linalg.norm(s1 - s0)),
        "delta_surface_inf": float(np.linalg.norm(s1 - s0, ord=np.inf)),
        "before": before,
        "after": after,
        "coil_x_delta_inf": float(
            np.linalg.norm(np.asarray(biot_savart.x) - coil0, ord=np.inf)
        ),
        "nphi": int(np.asarray(working.quadpoints_phi).size),
        "ntheta": int(np.asarray(working.quadpoints_theta).size),
        "purpose_close": bool(
            success or after["masked_l2"] < EXACT_PURPOSE_RESID
        ),
        "match_nested_exact": bool(success and persist),
    }


def _candidate_from_lane(path: Path) -> tuple[Fullspace675Candidate, dict[str, object]]:
    lane = json.loads(path.read_bytes())
    block = lane["result"]["endpoint_candidate"]
    candidate = Fullspace675Candidate(
        coil_coordinates=tuple(float(v) for v in block["coil_coordinates"]),
        vessel_coordinates=tuple(float(v) for v in block["vessel_coordinates"]),
        surface_coordinates=tuple(float(v) for v in block["surface_coordinates"]),
    )
    return candidate, {
        "lane": lane.get("lane"),
        "endpoint_inner_state": lane["result"].get("endpoint_inner_state"),
        "endpoint_objective": lane["result"].get("objective_value"),
        "path": str(path),
    }


def _public_ls(result: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def _try_point(
    name: str,
    candidate: Fullspace675Candidate,
    materializer: Fullspace675NativeBoozerSystemMaterializer,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    _log(f"\n=== {name}: materialize ===")
    t0 = time.perf_counter()
    physical = materializer.materialize_physical_state(candidate)
    qr = _qr_inner(physical.system.design_matrix, physical.system.right_hand_side)
    _log(
        f"{name} QR iota={qr['iota']:.16f} G={qr['G']:.16f} "
        f"resid={qr['residual_l2']:.6e} in {time.perf_counter() - t0:.2f}s"
    )
    surface = physical.surface
    biot_savart = physical.biot_savart
    report: dict[str, object] = {
        "name": name,
        "qr_inner": qr,
        "source": extra,
        "nphi": int(np.asarray(surface.quadpoints_phi).size),
        "ntheta": int(np.asarray(surface.quadpoints_theta).size),
    }

    _log(f"=== {name}: LS Newton ===")
    ls = _run_ls_newton(biot_savart, surface, qr["iota"], qr["G"])
    _log(
        f"{name} LS success={ls['success']} iter={ls['iter']} "
        f"match={ls['match_nested_ls']} purpose_close={ls['purpose_close']} "
        f"d_iota={ls['delta_iota']:.6e} ||ds||_inf={ls['delta_surface_inf']:.6e} "
        f"grad {ls['grad_inf_before']:.3e}->{ls['grad_inf_after']:.3e}"
    )
    report["ls"] = _public_ls(ls)

    _log(f"=== {name}: exact from un-nest s ===")
    try:
        exact_from_unnest = _run_exact_newton(
            biot_savart, surface, qr["iota"], qr["G"], seed="unnest_s"
        )
        report["exact_from_unnest"] = exact_from_unnest
        _log(
            f"{name} exact-unnest success={exact_from_unnest['success']} "
            f"persist={exact_from_unnest['persisted']} iter={exact_from_unnest['iter']} "
            f"masked {exact_from_unnest['before']['masked_l2']:.3e}->"
            f"{exact_from_unnest['after']['masked_l2']:.3e} "
            f"min={exact_from_unnest['min_masked_l2']:.3e}"
        )
    except Exception as exc:  # noqa: BLE001
        report["exact_from_unnest"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _log(f"{name} exact-unnest FAILED: {exc}")

    _log(f"=== {name}: LS→exact handoff ===")
    try:
        exact_from_ls = _run_exact_newton(
            biot_savart,
            ls["_working"],
            ls["iota"],
            ls["G"],
            seed="ls_polished_s",
        )
        report["exact_from_ls_handoff"] = exact_from_ls
        _log(
            f"{name} exact-handoff success={exact_from_ls['success']} "
            f"persist={exact_from_ls['persisted']} iter={exact_from_ls['iter']} "
            f"masked {exact_from_ls['before']['masked_l2']:.3e}->"
            f"{exact_from_ls['after']['masked_l2']:.3e} "
            f"min={exact_from_ls['min_masked_l2']:.3e} "
            f"d_iota={exact_from_ls['delta_iota']:.6e}"
        )
    except Exception as exc:  # noqa: BLE001
        report["exact_from_ls_handoff"] = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _log(f"{name} exact-handoff FAILED: {exc}")
    return report


def main() -> int:
    _log(f"repo={REPO}")
    _log(f"python={sys.executable}")
    validated = validate_frozen_genuine_675_input_bundle(
        BUNDLE_MANIFEST,
        source_repo=REPO,
    )
    material = Fullspace675NativeBoozerMaterial.from_runtime_spec(
        validated.runtime_spec,
        native_biot_savart_payload=validated.native_biot_savart_payload,
    )
    materializer = Fullspace675NativeBoozerSystemMaterializer(
        material=material,
        policy=validated.manifest.boozer_construction_policy,
    )
    f3_candidate, f3_meta = _candidate_from_lane(F3_B37_LANE)
    payload = {
        "analysis": "boozer_ls_and_exact_purpose_without_nested_journey",
        "bars": {
            "ls_match": {"delta_iota": LS_MATCH_IOTA, "delta_s_inf": LS_MATCH_S_INF},
            "ls_purpose_close": {
                "delta_iota": LS_PURPOSE_IOTA,
                "delta_s_inf": LS_PURPOSE_S_INF,
            },
            "exact_purpose_close_masked_l2": EXACT_PURPOSE_RESID,
            "exact_tol": EXACT_NEWTON_TOL,
        },
        "points": [
            _try_point("archived_start", validated.manifest.candidate, materializer),
            _try_point(
                "f3_b37_pair2_l1_endpoint",
                f3_candidate,
                materializer,
                f3_meta,
            ),
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _log(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
