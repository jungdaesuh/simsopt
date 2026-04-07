import numpy as np

from alm_utils import (
    augmented_inequality_objective,
    lower_bound_residual,
    upper_bound_residual,
)


def make_stage2_fun(JF, new_bs, new_surf, Jf, Jls, Jccdist, Jc):
    def fun(dofs):
        JF.x = dofs
        J = JF.J()
        grad = JF.dJ()
        unitn = new_surf.unitnormal()
        BdotN = np.mean(np.abs(np.sum(new_bs.B().reshape(unitn.shape) * unitn, axis=2)))
        outstr = f"J={J:.1e}, Jf={Jf.J():.1e}, ⟨B·n⟩={BdotN:.1e}"
        outstr += f", Len={Jls.J():.1f}m"
        outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}m"
        outstr += f", Curvature={Jc.J():.2f}"
        outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
        print(outstr)
        return J, grad

    return fun


def evaluate_stage2_hardware_constraints(
    coil_length,
    length_target,
    curve_curve_min_dist,
    cc_threshold,
    max_curvature,
    curvature_threshold,
):
    violations = []
    if coil_length > length_target:
        violations.append(
            f"coil_length {coil_length:.6f} exceeds target {length_target:.6f}"
        )
    if curve_curve_min_dist < cc_threshold:
        violations.append(
            f"coil_coil_min_dist {curve_curve_min_dist:.6f} below threshold {cc_threshold:.6f}"
        )
    if max_curvature > curvature_threshold:
        violations.append(
            f"max_curvature {max_curvature:.6f} exceeds threshold {curvature_threshold:.6f}"
        )
    return {
        "success": len(violations) == 0,
        "violations": violations,
        "coil_length": float(coil_length),
        "length_target": float(length_target),
        "curve_curve_min_dist": float(curve_curve_min_dist),
        "cc_threshold": float(cc_threshold),
        "max_curvature": float(max_curvature),
        "curvature_threshold": float(curvature_threshold),
    }


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
    stage2_constraint_activity_tolerances,
    smooth_min_distance_signed_constraint,
    smooth_max_curvature_signed_constraint,
):
    base_objective.x = dofs
    base_value = float(base_objective.J())
    base_grad = np.asarray(base_objective.dJ(), dtype=float)
    base_objective_optimizable = base_objective

    coil_length = float(Jls.J())
    length_violation = upper_bound_residual(coil_length, length_target)
    length_grad = np.asarray(Jls.dJ(partials=True)(base_objective_optimizable), dtype=float)

    curve_curve_min_dist = float(Jccdist.shortest_distance())
    curve_curve_violation = lower_bound_residual(
        curve_curve_min_dist,
        Jccdist.minimum_distance,
    )
    curve_curve_signed_value, curve_curve_grad = smooth_min_distance_signed_constraint(
        Jccdist.curves,
        Jccdist.minimum_distance,
        distance_smoothing,
        base_objective_optimizable,
    )

    max_curvature = float(np.max(Jc.curve.kappa()))
    curvature_violation = upper_bound_residual(max_curvature, Jc.threshold)
    curvature_signed_value, curvature_grad = smooth_max_curvature_signed_constraint(
        Jc.curve,
        Jc.threshold,
        curvature_smoothing,
        base_objective_optimizable,
    )

    evaluation = augmented_inequality_objective(
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
            "constraint_names": [
                "coil_length_upper_bound",
                "coil_coil_spacing",
                "max_curvature",
            ],
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
        }
    )

    unitn = new_surf.unitnormal()
    BdotN = np.mean(np.abs(np.sum(new_bs.B().reshape(unitn.shape) * unitn, axis=2)))
    outstr = (
        f"ALM J={evaluation['total']:.1e}, Jflux={base_value:.1e}, "
        f"Jf={Jf.J():.1e}, ⟨B·n⟩={BdotN:.1e}"
    )
    outstr += (
        f", Len={coil_length:.1f}m, Len+={length_violation:.2e}, "
        f"Leng={coil_length - length_target:.2e}"
    )
    outstr += (
        f", C-C-Sep={curve_curve_min_dist:.2f}m, CC+={curve_curve_violation:.2e}, "
        f"CCg={curve_curve_signed_value:.2e}"
    )
    outstr += (
        f", Curvature={max_curvature:.2f}, Curv+={curvature_violation:.2e}, "
        f"Curvg={curvature_signed_value:.2e}"
    )
    outstr += f", ║∇L_A║={evaluation['stationarity_norm']:.1e}, μ={penalty:.1e}"
    print(outstr)
    return evaluation
