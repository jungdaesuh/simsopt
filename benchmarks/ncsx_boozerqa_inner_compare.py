#!/usr/bin/env python
"""Serial NCSX boozerQA inner compare: native run_code, JAX run_code, JAX Schur.

Not a sealed claim. Isolation: one lane per process. Native uses banana
``run_code``; JAX run_code uses the LS Newton dense-LU default; JAX Schur
uses reduced nested-LS inner. Do not inherit F3 7.70×.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(REPO / "src")

import numpy as np
from scipy.optimize import minimize
from simsopt._core import load
from simsopt.field import BiotSavart
from simsopt.geo import Volume
from simsopt.geo.boozersurface import BoozerSurface
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NestedLsOuterAcceptWithoutCandidate,
    NestedLsOuterCandidateStore,
    nested_ls_outer_rejection_barrier,
)
from simsopt_jax_adapters.geo.nested_ls_ncsx import (
    NCSX_EXAMPLE_JSON,
    NcsxNestedLsBranchJump,
    NcsxNestedLsInnerSolveFailed,
    NcsxNestedLsProblem,
    NcsxNestedLsSelfIntersecting,
    clone_surface_xyz_tensor_fourier,
    commit_ncsx_anchor,
    ncsx_nested_ls_outer_value_and_grad,
    ncsx_problem_from_jax_boozers,
    restore_ncsx_anchor,
    run_ncsx_schur_inner,
    upsample_surface_xyz_tensor_fourier,
)

DEFAULT_OUT = REPO / ".artifacts" / "ncsx_boozerqa_inner_compare.json"
# Containment barrier coefficient for infeasible trials. Same number the
# sealed B3 outer uses; this diagnostic does not inherit F3 ftol/gtol/maxls.
NCSX_REJECTION_DISTANCE_SCALE = 1.0


def _stats(xs):
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


def _time(fn):
    started = time.perf_counter()
    result = fn()
    return float(time.perf_counter() - started), result


def _summarize_run_code(out) -> dict[str, object]:
    return {
        "success": bool(out["success"]),
        "iter": int(out.get("iter", -1)),
        "iota": float(out["iota"]),
        "G": float(out["G"]),
    }


def _summarize_schur(out) -> dict[str, object]:
    return {
        "success": bool(out.success),
        "exit_status": str(out.exit_status),
        "iter": int(out.iteration_count),
        "iota": float(out.iota),
        "G": float(out.G),
        "grad_l2": float(np.linalg.norm(out.reduced_gradient)),
    }


def _scale_ncsx_surface(old, *, mpol: int, ntor: int, nphi: int, ntheta: int):
    if mpol == old.mpol and ntor == old.ntor and nphi == len(old.quadpoints_phi):
        return clone_surface_xyz_tensor_fourier(old)
    return upsample_surface_xyz_tensor_fourier(
        old, mpol=mpol, ntor=ntor, nphi=nphi, ntheta=ntheta
    )


def _load_ncsx(
    *,
    mpol: int | None,
    ntor: int | None,
    nphi: int,
    ntheta: int,
    nsurfaces: int = 1,
):
    packed = load(str(NCSX_EXAMPLE_JSON))
    base_curves, base_currents, coils, curves, surfaces, boozer_surfaces, ress = packed
    nsurf = int(nsurfaces)
    if nsurf < 1 or nsurf > len(surfaces):
        raise ValueError(
            f"nsurfaces must be in [1, {len(surfaces)}]; got {nsurf}."
        )
    old0 = surfaces[0]
    if mpol is None:
        mpol = int(old0.mpol)
    if ntor is None:
        ntor = int(old0.ntor)
    scaled = [
        _scale_ncsx_surface(
            old, mpol=mpol, ntor=ntor, nphi=nphi, ntheta=ntheta
        )
        for old in surfaces[:nsurf]
    ]
    return {
        "base_curves": base_curves,
        "base_currents": base_currents,
        "coils": coils,
        "curves": curves,
        "surface": scaled[0],
        "surfaces": scaled,
        "iotas": [float(res["iota"]) for res in ress[:nsurf]],
        "gs": [float(res["G"]) for res in ress[:nsurf]],
        "constraint_weight": float(boozer_surfaces[0].constraint_weight),
        "iota0": float(ress[0]["iota"]),
        "g0": float(ress[0]["G"]),
        "mpol": mpol,
        "ntor": ntor,
        "nsurfaces": nsurf,
    }


def _run_repeats(fn, repeats: int):
    cold_s, cold_r = _time(fn)
    warm = []
    last = cold_r
    for _ in range(repeats):
        seconds, result = _time(fn)
        warm.append(seconds)
        last = result
    return {
        "cold_s": cold_s,
        "cold": last if not warm else cold_r,
        "warm_s": _stats(warm) if warm else None,
        "warm": last,
    }


def _lane_native(problem, *, bfgs_maxiter: int, repeats: int):
    surf = clone_surface_xyz_tensor_fourier(problem["surface"])
    vol = Volume(surf)
    native = BoozerSurface(
        BiotSavart(problem["coils"]),
        surf,
        vol,
        float(vol.J()),
        constraint_weight=problem["constraint_weight"],
        options={
            "verbose": False,
            "bfgs_tol": 1e-10,
            "newton_tol": 1e-11,
            "newton_maxiter": 40,
            "bfgs_maxiter": int(bfgs_maxiter),
            "weight_inv_modB": True,
        },
    )
    land_s, land_r = _time(
        lambda: _summarize_run_code(
            native.minimize_boozer_penalty_constraints_newton(
                iota=problem["iota0"],
                G=problem["g0"],
                verbose=False,
                maxiter=40,
                tol=1e-11,
                constraint_weight=problem["constraint_weight"],
            )
        )
    )

    def native_run():
        native.need_to_run_code = True
        return _summarize_run_code(native.run_code(problem["iota0"], problem["g0"]))

    timed = _run_repeats(native_run, repeats)
    timed["newton_land"] = {"seconds": land_s, "result": land_r}
    return timed


def _configure_jax_gpu() -> None:
    os.environ.setdefault("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")


def _lane_jax_runcode(problem, *, bfgs_maxiter: int, repeats: int):
    _configure_jax_gpu()
    import jax

    jax.config.update("jax_enable_x64", True)
    surf = clone_surface_xyz_tensor_fourier(problem["surface"])
    vol = Volume(surf)
    jax_bz = BoozerSurfaceJAX(
        BiotSavartJAX(problem["coils"]),
        surf,
        vol,
        float(vol.J()),
        constraint_weight=problem["constraint_weight"],
        options={
            "verbose": False,
            "optimizer_backend": "ondevice",
            "bfgs_tol": 1e-10,
            "newton_tol": 1e-11,
            "newton_maxiter": 40,
            "bfgs_maxiter": int(bfgs_maxiter),
            "weight_inv_modB": True,
        },
    )
    payload = {
        "jax_backend": str(jax.default_backend()),
        "jax_devices": [str(device) for device in jax.devices()],
        "newton_linear_solver": jax_bz.options["newton_linear_solver"],
    }

    def jax_run():
        jax_bz.need_to_run_code = True
        return _summarize_run_code(jax_bz.run_code(problem["iota0"], problem["g0"]))

    payload.update(_run_repeats(jax_run, repeats))
    return payload


def _lane_jax_schur(problem, *, repeats: int):
    _configure_jax_gpu()
    import jax

    jax.config.update("jax_enable_x64", True)
    surf = clone_surface_xyz_tensor_fourier(problem["surface"])
    vol = Volume(surf)
    jax_bz = BoozerSurfaceJAX(
        BiotSavartJAX(problem["coils"]),
        surf,
        vol,
        float(vol.J()),
        constraint_weight=problem["constraint_weight"],
        options={
            "verbose": False,
            "optimizer_backend": "ondevice",
            "weight_inv_modB": True,
        },
    )
    payload = {
        "jax_backend": str(jax.default_backend()),
        "jax_devices": [str(device) for device in jax.devices()],
    }

    def jax_run():
        return _summarize_schur(
            run_ncsx_schur_inner(
                jax_bz,
                iota=problem["iota0"],
                G=problem["g0"],
            )
        )

    payload.update(_run_repeats(jax_run, repeats))
    return payload


@dataclass
class _NcsxOuterCandidate:
    """One feasible NCSX outer evaluation, pending scipy acceptance."""

    j: float
    gradient: np.ndarray
    coil_dofs: np.ndarray
    surface_dofs: tuple[np.ndarray, ...]
    iotas: tuple[float, ...]
    gs: tuple[float, ...]


def _snapshot_ncsx_candidate(
    nested: NcsxNestedLsProblem,
    *,
    value: float,
    gradient: np.ndarray,
    coil_dofs: np.ndarray,
) -> _NcsxOuterCandidate:
    return _NcsxOuterCandidate(
        j=float(value),
        gradient=np.array(gradient, dtype=np.float64, copy=True),
        coil_dofs=np.array(coil_dofs, dtype=np.float64, copy=True),
        surface_dofs=tuple(
            np.array(
                surface_state.jax_boozer.surface.get_dofs(),
                dtype=np.float64,
                copy=True,
            )
            for surface_state in nested.surfaces
        ),
        iotas=tuple(float(surface_state.iota) for surface_state in nested.surfaces),
        gs=tuple(float(surface_state.G) for surface_state in nested.surfaces),
    )


def _install_ncsx_candidate(
    nested: NcsxNestedLsProblem, candidate: _NcsxOuterCandidate
) -> None:
    for surface_state, sdofs, iota, g_value in zip(
        nested.surfaces,
        candidate.surface_dofs,
        candidate.iotas,
        candidate.gs,
        strict=True,
    ):
        surface_state.jax_boozer.surface.set_dofs(
            np.array(sdofs, dtype=np.float64, copy=True)
        )
        surface_state.iota = float(iota)
        surface_state.G = float(g_value)
    commit_ncsx_anchor(nested)


def _summarize_committed(candidates: NestedLsOuterCandidateStore[_NcsxOuterCandidate]):
    if not candidates.is_primed:
        return None
    incumbent = candidates.committed
    return {
        "j": float(incumbent.j),
        "iotas": [float(value) for value in incumbent.iotas],
        "gs": [float(value) for value in incumbent.gs],
        "coil_dofs": int(incumbent.coil_dofs.size),
    }


def _lane_jax_outer(problem, *, maxiter: int):
    _configure_jax_gpu()
    import jax

    jax.config.update("jax_enable_x64", True)
    # Match boozerQA_ls_mpi.py: freeze the first current so overall current
    # scale is not a free outer DOF.
    problem["base_currents"][0].fix_all()
    landers = []
    landed_iotas = []
    landed_gs = []
    for surface, iota0, g0 in zip(
        problem["surfaces"], problem["iotas"], problem["gs"], strict=True
    ):
        land_surface = clone_surface_xyz_tensor_fourier(surface)
        land_label = Volume(land_surface)
        lander = BoozerSurfaceJAX(
            BiotSavartJAX(problem["coils"]),
            land_surface,
            land_label,
            float(land_label.J()),
            constraint_weight=problem["constraint_weight"],
            options={
                "verbose": False,
                "optimizer_backend": "ondevice",
                "bfgs_maxiter": 20,
                "weight_inv_modB": True,
            },
        )
        lander.need_to_run_code = True
        landed = lander.run_code(iota0, g0)
        if landed is None or not bool(landed["success"]):
            return {
                "seconds": 0.0,
                "success": False,
                "message": "banana run_code land for NCSX outer failed",
                "nsurfaces": int(problem["nsurfaces"]),
                "last_inner": None,
            }
        landers.append(lander)
        landed_iotas.append(float(landed["iota"]))
        landed_gs.append(float(landed["G"]))
    nested = ncsx_problem_from_jax_boozers(
        landers,
        iotas=landed_iotas,
        g_values=landed_gs,
        base_curves=problem["base_curves"],
        curves=problem["curves"],
    )
    x0 = np.array(nested.biotsavart.x, dtype=np.float64, copy=True)
    candidates = NestedLsOuterCandidateStore[_NcsxOuterCandidate](x0)
    rejection_scale = float(NCSX_REJECTION_DISTANCE_SCALE)
    nfev = 0
    n_rejected = 0

    def fun(dofs):
        nonlocal nfev, n_rejected
        nfev += 1
        point = np.array(dofs, dtype=np.float64, copy=True)
        try:
            value, gradient = ncsx_nested_ls_outer_value_and_grad(nested, point)
        except (
            NcsxNestedLsInnerSolveFailed,
            NcsxNestedLsBranchJump,
            NcsxNestedLsSelfIntersecting,
        ) as signal:
            restore_ncsx_anchor(nested)
            if not candidates.is_primed:
                raise
            if candidates.committed_matches(point):
                raise RuntimeError(
                    "the nested inner solve rejected the exact committed outer "
                    "point; no line-search barrier can represent that failure"
                ) from signal
            n_rejected += 1
            incumbent = candidates.committed
            sentinel, sentinel_grad = nested_ls_outer_rejection_barrier(
                anchor_value=incumbent.j,
                anchor_parameters=incumbent.coil_dofs,
                trial_parameters=point,
                scale=rejection_scale,
            )
            return float(sentinel), np.asarray(sentinel_grad, dtype=np.float64)
        candidate = _snapshot_ncsx_candidate(
            nested, value=value, gradient=gradient, coil_dofs=point
        )
        candidates.record(point, candidate)
        _install_ncsx_candidate(nested, candidates.committed)
        return float(value), np.asarray(gradient, dtype=np.float64)

    def callback(xk):
        committed = candidates.accept(np.asarray(xk, dtype=np.float64))
        _install_ncsx_candidate(nested, committed)

    started = time.perf_counter()
    try:
        result = minimize(
            fun,
            x0,
            method="L-BFGS-B",
            jac=True,
            callback=callback,
            options={"maxiter": int(maxiter), "maxcor": 20},
        )
    except (
        NestedLsOuterAcceptWithoutCandidate,
        NcsxNestedLsInnerSolveFailed,
        NcsxNestedLsBranchJump,
        NcsxNestedLsSelfIntersecting,
    ) as error:
        seconds = float(time.perf_counter() - started)
        return {
            "seconds": seconds,
            "success": False,
            "message": str(error),
            "nfev": nfev,
            "n_rejected": n_rejected,
            "rejection_distance_scale": rejection_scale,
            "coil_dofs": int(x0.size),
            "nsurfaces": int(problem["nsurfaces"]),
            "committed": _summarize_committed(candidates),
            "last_inner": None
            if nested.last_inner is None
            else _summarize_schur(nested.last_inner),
        }
    seconds = float(time.perf_counter() - started)
    return {
        "seconds": seconds,
        "success": bool(result.success),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "fun_evals": nfev,
        "n_rejected": n_rejected,
        "rejection_distance_scale": rejection_scale,
        "coil_dofs": int(x0.size),
        "nsurfaces": int(problem["nsurfaces"]),
        "fun": float(result.fun),
        "message": str(result.message),
        "committed": _summarize_committed(candidates),
        "last_inner": None
        if nested.last_inner is None
        else _summarize_schur(nested.last_inner),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lane",
        choices=("native", "jax-runcode", "jax-schur", "jax-outer", "all"),
        default="all",
    )
    parser.add_argument("--mpol", type=int, default=None)
    parser.add_argument("--ntor", type=int, default=None)
    parser.add_argument("--nphi", type=int, default=18)
    parser.add_argument("--ntheta", type=int, default=18)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bfgs-maxiter", type=int, default=20)
    parser.add_argument("--outer-maxiter", type=int, default=2)
    parser.add_argument(
        "--nsurfaces",
        type=int,
        default=1,
        help="Outer lane surface count (boozerQA_ls_mpi.py uses 2). Inner lanes use surface 0.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    loaded = _load_ncsx(
        mpol=args.mpol,
        ntor=args.ntor,
        nphi=args.nphi,
        ntheta=args.ntheta,
        nsurfaces=args.nsurfaces,
    )
    surface = loaded["surface"]
    payload: dict[str, object] = {
        "lane": args.lane,
        "mpol": loaded["mpol"],
        "ntor": loaded["ntor"],
        "nphi": len(surface.quadpoints_phi),
        "ntheta": len(surface.quadpoints_theta),
        "surface_dofs": int(surface.x.size),
        "nsurfaces": loaded["nsurfaces"],
        "constraint_weight": loaded["constraint_weight"],
        "diagnostic": True,
        "sealed_claim": False,
    }
    print(
        f"ncsx compare lane={args.lane} dofs={payload['surface_dofs']} "
        f"grid={payload['nphi']}x{payload['ntheta']}",
        flush=True,
    )
    lanes = (
        ("native", "jax-runcode", "jax-schur", "jax-outer")
        if args.lane == "all"
        else (args.lane,)
    )
    for lane in lanes:
        print(f"start {lane}", flush=True)
        if lane == "native":
            payload["native"] = _lane_native(
                loaded, bfgs_maxiter=args.bfgs_maxiter, repeats=args.repeats
            )
        elif lane == "jax-runcode":
            payload["jax_runcode"] = _lane_jax_runcode(
                loaded, bfgs_maxiter=args.bfgs_maxiter, repeats=args.repeats
            )
        elif lane == "jax-schur":
            payload["jax_schur"] = _lane_jax_schur(loaded, repeats=args.repeats)
        else:
            payload["jax_outer"] = _lane_jax_outer(
                loaded, maxiter=args.outer_maxiter
            )
        print(f"done {lane}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
