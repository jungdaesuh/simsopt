#!/usr/bin/env python3
"""Hot-path objective equivalence probe.

This script imports a requested SIMSOPT source tree explicitly and records
``J``/``dJ`` values for the objects covered by the hot-path performance plan.
Run it once against the original baseline checkout and once against the current
checkout, then compare the two JSON files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_REPO_SRC = (Path(__file__).resolve().parents[2] / "src").resolve()
DEFAULT_GRAD_REL_TOL = 1.0e-12
DEFAULT_GRAD_ABS_TOL = 1.0e-12
DEFAULT_J_ABS_TOL = 1.0e-12


def _activate_source_tree(source_root: Path) -> None:
    source_root = source_root.resolve()
    source_src = source_root / "src"
    examples_root = source_root / "examples" / "single_stage_optimization"
    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if finder.__class__.__module__ != "_simsopt_editable"
    ]
    sys.path = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() not in (SCRIPT_REPO_SRC, source_src)
    ]
    sys.path.insert(0, str(examples_root))
    sys.path.insert(0, str(source_src))
    os.chdir(source_root)


def _curve_xyz(nquad: int = 48, order: int = 3):
    from simsopt.geo import CurveXYZFourier

    curve = CurveXYZFourier(np.linspace(0.0, 1.0, nquad, endpoint=False), order=order)
    curve.x = np.zeros(curve.dof_size)
    curve.set("xc(0)", 0.93)
    curve.set("xc(1)", 0.11)
    curve.set("ys(1)", 0.10)
    curve.set("zs(1)", 0.055)
    if order >= 2:
        curve.set("xc(2)", 0.018)
        curve.set("ys(2)", -0.012)
        curve.set("zs(2)", 0.021)
    return curve


def _flat_array(value) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _record(records: dict[str, dict[str, object]], name: str, objective) -> None:
    gradient_value = objective.dJ()
    gradient = _flat_array(gradient_value)
    records[name] = {
        "J": float(objective.J()),
        "dJ": gradient.tolist(),
        "dJ_shape": list(np.asarray(gradient_value, dtype=float).shape),
        "dJ_norm": float(np.linalg.norm(gradient)),
        "dJ_maxabs": float(np.max(np.abs(gradient))) if gradient.size else 0.0,
    }


def _record_pair(
    records: dict[str, dict[str, object]],
    name: str,
    objective_value: float,
    gradient_value,
) -> None:
    gradient = _flat_array(gradient_value)
    records[name] = {
        "J": float(objective_value),
        "dJ": gradient.tolist(),
        "dJ_shape": list(np.asarray(gradient_value, dtype=float).shape),
        "dJ_norm": float(np.linalg.norm(gradient)),
        "dJ_maxabs": float(np.max(np.abs(gradient))) if gradient.size else 0.0,
    }


def _force_objectives(records: dict[str, dict[str, object]]) -> None:
    from simsopt.configs import get_ncsx_data
    from simsopt.field import Coil
    from simsopt.field.force import LpCurveForce, MeanSquaredForce
    from simsopt.field.selffield import regularization_circ

    curves, currents, _ = get_ncsx_data(Nt_coils=2)
    coils = [Coil(curve, current) for curve, current in zip(curves, currents)]
    regularization = regularization_circ(0.05)
    _record(records, "MeanSquaredForce", MeanSquaredForce(coils[0], coils, regularization))
    _record(records, "LpCurveForce", LpCurveForce(coils[0], coils, regularization, 2.5))


def _curve_objectives(records: dict[str, dict[str, object]]) -> None:
    from simsopt.field import Current, coils_via_symmetries
    from simsopt.geo import (
        SurfaceRZFourier,
        create_equally_spaced_curves,
        create_multifilament_grid,
    )
    from simsopt.geo.curveobjectives import (
        CurveCurveDistance,
        CurveLength,
        CurveSurfaceDistance,
    )

    base_curves = create_equally_spaced_curves(
        3, 2, True, R0=0.9, R1=0.22, order=2, numquadpoints=48
    )
    base_currents = [Current(1.0e4) for _ in base_curves]
    curves = [
        coil.curve for coil in coils_via_symmetries(base_curves, base_currents, 2, True)
    ]
    _record(records, "CurveCurveDistance", CurveCurveDistance(curves, 2.0, num_basecurves=3))

    surface = SurfaceRZFourier.from_nphi_ntheta(nfp=2, nphi=16, ntheta=16, ntor=0)
    surface.set("rc(0,0)", 1.05)
    surface.set("rc(1,0)", 0.18)
    surface.set("zs(1,0)", 0.18)
    _record(records, "CurveSurfaceDistance", CurveSurfaceDistance(curves, surface, 0.25))

    filament = create_multifilament_grid(
        base_curves[0], 2, 2, 0.01, 0.012, rotation_order=1, frame="centroid"
    )[0]
    _record(records, "CurveLength_CurveFilament_centroid", CurveLength(filament))


def _strain_objectives(records: dict[str, dict[str, object]]) -> None:
    from simsopt.configs import get_ncsx_data
    from simsopt.geo import (
        FrameRotation,
        FramedCurveCentroid,
        FramedCurveFrenet,
        ZeroRotation,
    )
    from simsopt.geo.strain_optimization import (
        LPBinormalCurvatureStrainPenalty,
        LPTorsionalStrainPenalty,
    )

    curves, _, _ = get_ncsx_data(Nt_coils=3, ppp=40)
    curve = curves[0]
    rotation = FrameRotation(curve.quadpoints, order=1)
    rotation.x = np.array([0.0, 0.1, 0.3])
    framed_curves = [
        ("centroid", FramedCurveCentroid(curve, rotation)),
        ("frenet", FramedCurveFrenet(curve, ZeroRotation(curve.quadpoints))),
    ]
    for frame_name, framed_curve in framed_curves:
        _record(
            records,
            f"LPBinormalCurvatureStrainPenalty_{frame_name}",
            LPBinormalCurvatureStrainPenalty(
                framed_curve, width=1.0e-3, p=2, threshold=1.0e-4
            ),
        )
        _record(
            records,
            f"LPTorsionalStrainPenalty_{frame_name}",
            LPTorsionalStrainPenalty(
                framed_curve, width=1.0e-3, p=2, threshold=1.0e-8
            ),
        )


def _banana_objectives(records: dict[str, dict[str, object]]) -> None:
    from banana_opt.ellipse_width import ProjectedEllipseWidth
    from banana_opt.fold_buildability import (
        CurveSurfaceGeodesicCurvature,
        NormalizedCurveCurvatureHinge,
        RotationAwareCurvatureExcessPenalty,
    )
    from banana_opt.hardware_keepout import CurveHardwareKeepout
    from banana_opt.poloidal_extent import PoloidalExtent
    from banana_opt.self_intersect import CurveSelfIntersect
    from simsopt.geo import FrameRotation, FramedCurveSurfaceTangent

    curve = _curve_xyz(48, 2)
    _record(
        records,
        "CurveSelfIntersect",
        CurveSelfIntersect(curve, minimum_distance=0.08, neighbor_skip=2, normalize=False),
    )

    rotation = FrameRotation(curve.quadpoints, order=1)
    rotation.x = np.array([0.15, 0.05, -0.02])
    framed_curve = FramedCurveSurfaceTangent(curve, 0.903, 0.0, rotation)
    _record(
        records,
        "CurveSurfaceGeodesicCurvature",
        CurveSurfaceGeodesicCurvature(framed_curve, p=2, threshold=0.0),
    )
    _record(
        records,
        "NormalizedCurveCurvatureHinge",
        NormalizedCurveCurvatureHinge(curve, p=4, threshold=2.0),
    )
    _record(records, "PoloidalExtent", PoloidalExtent(curve, 0.903, 0.5))
    _record(records, "ProjectedEllipseWidth", ProjectedEllipseWidth(curve, 0.903, 0.142))

    hardware_points = np.array(
        [[1.05, 0.0, 0.0], [0.96, 0.08, 0.015], [10.0, 0.0, 0.0]]
    )
    _record(
        records,
        "CurveHardwareKeepout",
        CurveHardwareKeepout(
            [curve],
            hardware_points,
            minimum_distance=0.04,
            point_weight=1.0e-4,
            winding_r0=0.0,
        ),
    )
    _record(
        records,
        "RotationAwareCurvatureExcessPenalty",
        RotationAwareCurvatureExcessPenalty(framed_curve, curve, finite_build=None, p=2),
    )


def _biotsavart_vjps(records: dict[str, dict[str, object]]) -> None:
    from simsopt.field import BiotSavart, Coil, Current

    curve = _curve_xyz(36, 3)
    coil = Coil(curve, Current(1.0e4))
    biot_savart = BiotSavart([coil])
    points = np.asarray(
        9 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]], dtype=float
    )
    points += 1.0e-3 * np.linspace(-0.5, 0.5, points.size).reshape(points.shape)
    biot_savart.set_points(points)
    B = biot_savart.B()
    dB = biot_savart.dB_by_dX()
    A = biot_savart.A()
    dA = biot_savart.dA_by_dX()
    _record_pair(
        records,
        "BiotSavart_B_vjp_curve",
        float(np.sum(B * B)),
        biot_savart.B_vjp(B)(curve),
    )
    _record_pair(
        records,
        "BiotSavart_B_and_dB_vjp_curve",
        float(np.sum(dB * dB)),
        biot_savart.B_and_dB_vjp(B, dB)[1](curve),
    )
    _record_pair(
        records,
        "BiotSavart_A_vjp_coil",
        float(np.sum(A * A)),
        biot_savart.A_vjp(A)(coil),
    )
    _record_pair(
        records,
        "BiotSavart_A_and_dA_vjp_coil",
        float(np.sum(dA * dA)),
        biot_savart.A_and_dA_vjp(A, dA)[1](coil),
    )


def collect_records(source_root: Path) -> dict[str, dict[str, object]]:
    _activate_source_tree(source_root)
    records: dict[str, dict[str, object]] = {}
    _force_objectives(records)
    _curve_objectives(records)
    _strain_objectives(records)
    _banana_objectives(records)
    _biotsavart_vjps(records)
    return records


def _max_relative_delta(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    delta = np.abs(left - right)
    max_abs = float(np.max(delta)) if delta.size else 0.0
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0e-300)
    max_rel = float(np.max(delta / scale)) if delta.size else 0.0
    return max_abs, max_rel


def compare_records(
    baseline: dict[str, dict[str, object]],
    current: dict[str, dict[str, object]],
    *,
    grad_rel_tol: float,
    grad_abs_tol: float,
    j_abs_tol: float,
) -> list[str]:
    failures: list[str] = []
    for name in sorted(set(baseline) | set(current)):
        if name not in baseline or name not in current:
            failures.append(f"{name}: missing baseline={name in baseline} current={name in current}")
            continue
        base_record = baseline[name]
        current_record = current[name]
        if base_record["dJ_shape"] != current_record["dJ_shape"]:
            failures.append(
                f"{name}: gradient shape {base_record['dJ_shape']} != {current_record['dJ_shape']}"
            )
            continue
        base_j = float(base_record["J"])
        current_j = float(current_record["J"])
        base_grad = np.asarray(base_record["dJ"], dtype=float)
        current_grad = np.asarray(current_record["dJ"], dtype=float)
        grad_max_abs, grad_max_rel = _max_relative_delta(base_grad, current_grad)
        j_abs = abs(base_j - current_j)
        grad_ok = grad_max_rel <= grad_rel_tol or grad_max_abs <= grad_abs_tol
        if j_abs > j_abs_tol or not grad_ok:
            failures.append(
                f"{name}: J_abs={j_abs:.3e}, "
                f"dJ_max_abs={grad_max_abs:.3e}, dJ_max_rel={grad_max_rel:.3e}"
            )
    return failures


def _write_json(path: Path, records: dict[str, dict[str, object]]) -> None:
    path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


def _load_json(path: Path) -> dict[str, dict[str, object]]:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, help="source tree to probe")
    parser.add_argument("--json-out", type=Path, help="where to write probe JSON")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE_JSON", "CURRENT_JSON"),
        type=Path,
        help="compare two previously written probe outputs",
    )
    parser.add_argument("--grad-rel-tol", type=float, default=DEFAULT_GRAD_REL_TOL)
    parser.add_argument("--grad-abs-tol", type=float, default=DEFAULT_GRAD_ABS_TOL)
    parser.add_argument("--j-abs-tol", type=float, default=DEFAULT_J_ABS_TOL)
    args = parser.parse_args()

    if args.compare is not None:
        baseline = _load_json(args.compare[0])
        current = _load_json(args.compare[1])
        failures = compare_records(
            baseline,
            current,
            grad_rel_tol=args.grad_rel_tol,
            grad_abs_tol=args.grad_abs_tol,
            j_abs_tol=args.j_abs_tol,
        )
        print(
            f"Compared {len(set(baseline) | set(current))} records "
            f"with J absolute tolerance {args.j_abs_tol:.1e}, "
            f"dJ relative tolerance {args.grad_rel_tol:.1e}, "
            f"and dJ zero-component absolute tolerance {args.grad_abs_tol:.1e}."
        )
        if failures:
            for failure in failures:
                print(f"FAIL {failure}")
            return 1
        print("PASS hotpath equivalence probe")
        return 0

    if args.source_root is None or args.json_out is None:
        parser.error("--source-root and --json-out are required unless --compare is used")

    records = collect_records(args.source_root)
    _write_json(args.json_out, records)
    print(f"Wrote {len(records)} hotpath equivalence records to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
