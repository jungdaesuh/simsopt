"""Pure-Python tests for Taylor / FD gradient-correctness harness.

Validates the named tests from plan section "Gradient contract":

  * ``taylor_J2_alone_centered_eq28``     -- second-order, band [1.7, 2.2]
  * ``taylor_J_combined_forward_eq29_xcoils``  -- first-order, band [0.8, 1.2]
  * ``taylor_J_combined_centered_eq29_xsurface`` -- second-order, band [1.7, 2.2]
  * ``assert_dJ1_dxcoils_zero`` -- identity assertion

No VMEC dependency; uses analytic stubs whose third derivative is non-zero so
the centered-FD Taylor error is at the expected O(delta^2). A pure quadratic
stub would have centered FD exact (error at machine epsilon), defeating the
slope check — this is documented behavior, not a bug.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from simsopt._core.util import ObjectiveFailure
from simsopt.field import Current
from simsopt.field.coil import ScaledCurrent
from simsopt.objectives import QuadraticPenalty

from examples.single_stage_optimization.VMEC_SINGLE_STAGE import (
    FIRST_ORDER_SLOPE_BAND,
    SECOND_ORDER_SLOPE_BAND,
    assemble_dof_schema,
    assert_dJ1_dxcoils_zero,
    taylor_J2_alone_centered_eq28,
    taylor_J_combined_centered_eq29_xsurface,
    taylor_J_combined_forward_eq29_xcoils,
)
from examples.single_stage_optimization.VMEC_SINGLE_STAGE.vmec_single_stage_exceptions import (
    vmec_call_failure_from_objective_failure,
)
from examples.single_stage_optimization.VMEC_SINGLE_STAGE.vmec_single_stage_banana import (
    _BoundaryJ1FiniteDifferenceProblem,
    _CurrentMagnitude,
    _assign_vmec_schedule,
    _build_coil_block,
)


def _cubic_J_and_grad():
    """J(x) = sum(x^3), so H^3 J = 6 I (non-zero) -> centered FD has real O(delta^2) error."""
    def J(x: np.ndarray) -> float:
        return float(np.sum(x ** 3))

    def dJ(x: np.ndarray) -> np.ndarray:
        return 3.0 * x ** 2

    return J, dJ


def _quartic_J_and_grad():
    """J(x) = sum(x^4)/4; gradient = x^3; centered FD error is O(delta^2) from H^3 J != 0."""
    def J(x: np.ndarray) -> float:
        return float(np.sum(x ** 4) / 4.0)

    def dJ(x: np.ndarray) -> np.ndarray:
        return x ** 3

    return J, dJ


def _linear_J_and_grad(slope: np.ndarray):
    """J(x) = slope . x; gradient = slope; forward FD is exact (slope=0)
    BUT we want a function with non-zero second derivative so forward FD
    shows first-order behavior. Use J(x) = slope . x + 0.5 * x^T M x."""
    M = np.diag(np.arange(1.0, slope.size + 1.0))

    def J(x: np.ndarray) -> float:
        return float(slope @ x + 0.5 * x @ M @ x)

    def dJ(x: np.ndarray) -> np.ndarray:
        return slope + M @ x

    return J, dJ


def test_centered_fd_on_cubic_passes_second_order_band() -> None:
    J, dJ = _cubic_J_and_grad()
    x0 = np.array([1.0, 2.0, 3.0, 4.0])
    out = taylor_J2_alone_centered_eq28(J, dJ, x0)
    assert out["passed"] is True
    # Cubic has constant third derivative -> slope ≈ 2.0 across the full schedule.
    for slope in out["slopes"]:
        assert SECOND_ORDER_SLOPE_BAND[0] <= slope <= SECOND_ORDER_SLOPE_BAND[1]


def test_centered_fd_records_full_diagnostics() -> None:
    J, dJ = _cubic_J_and_grad()
    x0 = np.array([1.5, -2.5, 0.5])
    out = taylor_J2_alone_centered_eq28(J, dJ, x0)
    assert out["label"] == "taylor_J2_alone_centered_eq28"
    assert len(out["schedule"]) == len(out["errors"])
    assert len(out["slopes"]) == len(out["errors"]) - 1
    assert out["slope_band"] == (float(SECOND_ORDER_SLOPE_BAND[0]),
                                  float(SECOND_ORDER_SLOPE_BAND[1]))


def test_centered_fd_quartic_also_passes() -> None:
    J, dJ = _quartic_J_and_grad()
    x0 = np.array([1.0, -1.0, 2.0])
    out = taylor_J2_alone_centered_eq28(J, dJ, x0)
    assert out["passed"] is True


def test_forward_fd_on_quadratic_with_drift_passes_first_order_band() -> None:
    rng = np.random.default_rng(42)
    slope = rng.uniform(-1.0, 1.0, size=8)
    J, dJ = _linear_J_and_grad(slope)
    x0 = np.array([1.0, 0.5, -0.3, 2.0, 0.0, -1.5, 0.7, 0.1])
    # Coil slice covers the first 4 DOFs.
    out = taylor_J_combined_forward_eq29_xcoils(
        J, dJ, x0, coil_slice=slice(0, 4)
    )
    assert out["passed"] is True
    for slope in out["slopes"]:
        assert FIRST_ORDER_SLOPE_BAND[0] <= slope <= FIRST_ORDER_SLOPE_BAND[1]


def test_centered_fd_xsurface_on_cubic_passes() -> None:
    J, dJ = _cubic_J_and_grad()
    x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = taylor_J_combined_centered_eq29_xsurface(
        J, dJ, x0, surface_slice=slice(3, 6)
    )
    assert out["passed"] is True


def test_taylor_centered_xsurface_only_perturbs_surface_dofs() -> None:
    """The xsurface harness should leave coil DOFs untouched. Validated by
    asserting the direction vector zeros outside the surface slice."""
    # If the harness sweeps non-surface DOFs, the FD error would carry a coil
    # contribution. With a J that depends only on the second half (the
    # surface DOFs), an xsurface FD must converge correctly.
    def J(x: np.ndarray) -> float:
        return float(np.sum(x[3:] ** 3))

    def dJ(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        out[3:] = 3.0 * x[3:] ** 2
        return out

    x0 = np.array([100.0, 200.0, 300.0, 1.0, 2.0, 3.0])
    out = taylor_J_combined_centered_eq29_xsurface(
        J, dJ, x0, surface_slice=slice(3, 6)
    )
    assert out["passed"] is True


def test_assert_dJ1_dxcoils_zero_passes_on_constant_in_x_coils() -> None:
    """J1 stub depends only on surface DOFs; gradient over coil DOFs must be 0."""
    def J1(x: np.ndarray) -> float:
        return float(np.sum(x[3:] ** 2))

    x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # Coil slice is the first 3 DOFs. J1 is constant w.r.t. those.
    out = assert_dJ1_dxcoils_zero(J1, x0, coil_slice=slice(0, 3))
    assert out["passed"] is True
    assert out["max_abs_dJ1_dxcoils"] <= out["fd_zero_floor"]


def test_assert_dJ1_dxcoils_zero_fails_when_J1_depends_on_coils() -> None:
    """J1 stub depends on coil DOFs; identity check must report passed=False."""
    def J1_bad(x: np.ndarray) -> float:
        return float(np.sum(x ** 2))  # depends on all DOFs including coil slice

    x0 = np.array([1.0, 2.0, 3.0, 4.0])
    out = assert_dJ1_dxcoils_zero(J1_bad, x0, coil_slice=slice(0, 2))
    assert out["passed"] is False
    assert out["max_abs_dJ1_dxcoils"] > out["fd_zero_floor"]


def test_taylor_random_direction_is_reproducible_with_fixed_seed() -> None:
    J, dJ = _cubic_J_and_grad()
    x0 = np.array([1.0, 2.0, 3.0, 4.0])
    out1 = taylor_J2_alone_centered_eq28(J, dJ, x0, rng_seed=7)
    out2 = taylor_J2_alone_centered_eq28(J, dJ, x0, rng_seed=7)
    assert out1["errors"] == out2["errors"]


def test_taylor_different_seeds_give_different_directions() -> None:
    J, dJ = _cubic_J_and_grad()
    x0 = np.array([1.0, 2.0, 3.0, 4.0])
    out1 = taylor_J2_alone_centered_eq28(J, dJ, x0, rng_seed=0)
    out2 = taylor_J2_alone_centered_eq28(J, dJ, x0, rng_seed=1)
    # Different draws -> different errors at the same delta.
    assert out1["errors"] != out2["errors"]


def test_dof_schema_preserves_simsopt_coil_shape_name_format() -> None:
    """Coil-shape DOF names recorded in the schema must keep the SIMSOPT
    mode-name format (``xc(n)``, ``xs(n)``, ``yc(n)``, ``ys(n)``, ``zc(n)``,
    ``zs(n)`` produced by ``CurveXYZFourier.dof_names``). Asserting on the
    mode-pattern substrings ensures we did not silently fabricate names via
    a defensive fallback (see ``build_objective`` Fix A5)."""
    # Names as SIMSOPT emits them via ``Optimizable.dof_names``: class-tagged
    # prefix + colon + per-DOF token. Two coil-shape DOFs + one free current.
    simsopt_style_names = [
        "CurveXYZFourier1:xc(0)",
        "CurveXYZFourier1:ys(1)",
        "Current2:x0",
    ]
    schema = assemble_dof_schema(
        coil_dof_count=2,
        current_dof_count=1,
        boundary_dof_count=1,
        pinned_current_index=0,
        pinned_current_value_A=-8.0e4,
        fixed_boundary_modes=["rc(0,0)"],
        coils_currents_dof_names=simsopt_style_names,
        current_dof_names=["Current2:x0"],
        boundary_dof_names=["zs(1,1)"],
    )
    recorded = schema["coils_currents"]["dof_names_in_order"]
    assert recorded == simsopt_style_names
    assert schema["free_current_dof_names"] == ["Current2:x0"]
    # At least one coil-shape name must carry an ``[xyz][cs](`` token.
    mode_tokens = ("xc(", "xs(", "yc(", "ys(", "zc(", "zs(")
    coil_shape_names = recorded[: schema["coils_currents"]["coil_shape_count"]]
    assert any(
        any(tok in name for tok in mode_tokens) for name in coil_shape_names
    ), (
        "coil_shape dof_names lost SIMSOPT mode-pattern format; "
        "a defensive fallback may have re-entered build_objective"
    )


def test_build_coil_block_uses_fixed_tf_and_bounded_banana_seed() -> None:
    base_curves, base_currents, curves, coils = _build_coil_block(
        nfp=5,
        stellsym=True,
        tf_num=20,
        tf_R0=0.976,
        tf_R1=0.4,
        tf_order=1,
        tf_current_A=-8.0e4,
        winding_surface_R0=0.903,
        winding_surface_minor_radius=0.142,
        banana_order=2,
        banana_phi0=0.06,
        banana_phi1=0.03,
        banana_theta0=0.5,
        banana_theta1=0.1,
        banana_current_init_A=-1.0e4,
        nquadpoints=16,
    )

    assert len(base_curves) == 1
    assert len(base_currents) == 1
    assert len(coils) == 30
    assert len(curves) == 30
    assert base_currents[0].get_value() == pytest.approx(-1.0e4)
    assert len(base_currents[0].dof_names) == 1
    assert all(coil.current.get_value() == pytest.approx(-8.0e4) for coil in coils[:20])
    assert all(coil.current.dof_names == [] for coil in coils[:20])


def test_current_magnitude_penalty_is_zero_inside_cap_and_positive_above() -> None:
    inner_current = Current(1.0)
    banana_current = ScaledCurrent(inner_current, -1.0e4)
    penalty = QuadraticPenalty(_CurrentMagnitude(banana_current), 1.6e4, "max")

    assert penalty.J() == pytest.approx(0.0)
    inner_current.x = np.array([2.0])
    assert penalty.J() > 0.0


def test_objective_failure_ierr14_maps_to_nonfinite_output() -> None:
    failure = vmec_call_failure_from_objective_failure(
        ObjectiveFailure("VMEC did not converge. ierr=14"),
        phase="base",
    )

    assert failure.ier_flag == 14
    assert failure.failure_class == "vmec_nonfinite_output"
    assert failure.retryable is False
    assert str(failure) == "base VMEC failed: VMEC did not converge. ierr=14"


def test_boundary_j1_fd_problem_is_bound_optimizable_method() -> None:
    class FakeSurface:
        def __init__(self) -> None:
            self.x = np.array([1.0, -2.0])

    class FakeVmec:
        def __init__(self) -> None:
            self.need_to_run_code = False
            self.wout = SimpleNamespace(ns=2, iotaf=np.array([0.1, 0.2]))

        def aspect(self) -> float:
            return 12.7

        def volume(self) -> float:
            return 0.095

    fake_bundle = SimpleNamespace(
        surf=FakeSurface(),
        vmec=FakeVmec(),
        qs=SimpleNamespace(total=lambda: 0.0),
        config=SimpleNamespace(
            qs_weight=1.0,
            aspect_weight=1.0,
            aspect_target=12.7,
            iota_weight=1.0,
            iota_target=0.2,
            volume_weight=1.0,
            volume_target=0.095,
            iota_promotion_surface_s=1.0,
        ),
    )
    calls = []

    def run_vmec() -> None:
        calls.append(np.asarray(fake_bundle.surf.x, dtype=float).copy())

    problem = _BoundaryJ1FiniteDifferenceProblem(fake_bundle, run_vmec)
    problem.x = np.array([3.0, 4.0])

    assert problem.objective() == pytest.approx(0.0)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], np.array([3.0, 4.0]))
    assert fake_bundle.vmec.need_to_run_code is True


def test_assign_vmec_schedule_clears_stale_seed_tail() -> None:
    vi = SimpleNamespace(
        ns_array=np.array([13, 25, 51, 101]),
        ftol_array=np.array([1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10]),
        niter_array=np.array([10000, 10000, 10000, 10000]),
    )
    cfg = SimpleNamespace(
        ns_array=(13,),
        ftol_array=(1.0e-10,),
        niter_array=(3000,),
    )

    _assign_vmec_schedule(vi, cfg)

    np.testing.assert_array_equal(vi.ns_array, np.array([13, 0, 0, 0]))
    np.testing.assert_allclose(vi.ftol_array, np.array([1.0e-10, 0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(vi.niter_array, np.array([3000, 0, 0, 0]))
