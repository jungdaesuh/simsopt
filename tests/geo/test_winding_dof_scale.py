"""Coverage for the opt-in winding-size DOF variable transform (Phase 4 x_scale).

``run_scaled_winding_minimize`` applies u = x/scale around an L-BFGS-B solve so
the small-corridor winding SIZE dofs (rc(0,0)=R0, rc(1,0)/zs(1,0)=minor) are not
swamped by the stiffer curve harmonics. Default OFF (empty scale map -> all-ones)
must be byte-identical to a bare ``minimize``; when ON the chain rule, bounds
mapping, and result inversion must keep the optimum unchanged.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize

EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from banana_opt.single_stage_geometry import (  # noqa: E402
    build_curve_dof_scale_vector,
    build_sobolev_curve_mode_scale_vector,
    build_surface_dof_scale_vector,
    build_winding_dof_scale_vector,
    run_scaled_winding_minimize,
)
from banana_opt.stage2_geometry import WINDING_DOF_CORRIDOR_SCALE_MAP  # noqa: E402

# Path-qualified DOF names exactly as scipy's dof vector exposes them.
_NAMES = [
    "Current1:x0",
    "CurveCWSFourierCPP1:thetas(1)",
    "SurfaceRZFourier1:rc(0,0)",
    "SurfaceRZFourier1:rc(1,0)",
    "SurfaceRZFourier1:zs(1,0)",
]
_BOUNDS = [
    (-np.inf, np.inf),
    (-np.inf, np.inf),
    (0.903, 0.993),
    (0.130, 0.20),
    (0.130, 0.20),
]
_X0 = np.array([0.3, 0.1, 0.903, 0.142, 0.142])


def _separable_quadratic():
    """A separable quadratic with interior bounded minima. Mild conditioning so
    a tight-tolerance solve fully converges (the correctness tests below assert
    the transform preserves the minimizer, independent of the scale VALUE)."""
    c = np.array([50.0, 20.0, 3.0, 2.0, 2.0])
    t = np.array([0.4, 0.2, 0.95, 0.16, 0.16])

    def fun(x):
        d = np.asarray(x, dtype=float) - t
        return float(np.sum(c * d * d)), 2.0 * c * d

    return fun, t


def test_scale_vector_maps_size_suffixes_only():
    scale = build_winding_dof_scale_vector(_NAMES, WINDING_DOF_CORRIDOR_SCALE_MAP)
    assert scale[0] == 1.0  # banana current untouched
    assert scale[1] == 1.0  # curve harmonic untouched
    assert scale[2] == WINDING_DOF_CORRIDOR_SCALE_MAP["rc(0,0)"]
    assert scale[3] == WINDING_DOF_CORRIDOR_SCALE_MAP["rc(1,0)"]
    assert scale[4] == WINDING_DOF_CORRIDOR_SCALE_MAP["zs(1,0)"]


def test_empty_map_is_all_ones_identity():
    scale = build_winding_dof_scale_vector(_NAMES, {})
    assert np.array_equal(scale, np.ones(len(_NAMES)))


def test_nonpositive_scale_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="finite and positive"):
        build_winding_dof_scale_vector(_NAMES, {"rc(0,0)": 0.0})


def test_surface_scale_maps_surface_fourier_dofs_only():
    names = [
        "Current1:x0",
        "CurveCWSFourierCPP1:phic(1)",
        "SurfaceRZFourier1:rc(0,0)",
        "SurfaceRZFourier1:zs(1,-1)",
        "OtherObject:rc(0,0)",
    ]

    scale = build_surface_dof_scale_vector(names, surface_scale=1.0e-4)

    np.testing.assert_array_equal(
        scale,
        np.array([1.0, 1.0, 1.0e-4, 1.0e-4, 1.0]),
    )


def test_curve_scale_maps_curve_fourier_dofs_only():
    names = [
        "Current1:x0",
        "CurveCWSFourierCPP1:phic(1)",
        "CurveCWSFourierCPP1:thetas(7)",
        "SurfaceRZFourier1:rc(0,0)",
        "SurfaceRZFourier1:zs(1,0)",
    ]

    scale = build_curve_dof_scale_vector(names, curve_scale=0.25)

    np.testing.assert_array_equal(
        scale,
        np.array([1.0, 0.25, 0.25, 1.0, 1.0]),
    )


def test_curve_scale_rejects_invalid_values():
    import pytest

    for bad in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="finite and positive"):
            build_curve_dof_scale_vector(_NAMES, curve_scale=bad)


def test_default_off_is_byte_identical_to_plain_minimize():
    fun, _ = _separable_quadratic()
    scale = build_winding_dof_scale_vector(_NAMES, {})  # OFF -> all ones
    r_plain = minimize(fun, _X0, jac=True, method="L-BFGS-B", bounds=_BOUNDS)
    r_off = run_scaled_winding_minimize(minimize, fun, _X0, scale=scale, bounds=_BOUNDS)
    assert np.array_equal(r_off.x, r_plain.x)
    assert r_off.fun == r_plain.fun
    assert r_off.nit == r_plain.nit


def test_helper_scaled_objective_gradient_matches_central_fd():
    # Exercise the HELPER's real scaled objective (not a local reconstruction):
    # capture the (J, grad) closure run_scaled_winding_minimize hands the
    # optimizer, then FD-check that gradient. A helper that dropped the `* scale`
    # chain factor would return plain grad and fail here for the scaled dofs.
    fun, _ = _separable_quadratic()
    scale = build_winding_dof_scale_vector(_NAMES, WINDING_DOF_CORRIDOR_SCALE_MAP)
    captured = {}

    def capturing_minimize(scaled_fun, u0, **kwargs):
        captured["scaled_fun"] = scaled_fun
        return SimpleNamespace(
            x=np.asarray(u0, dtype=float),
            fun=0.0, nit=0, success=True, message="stub", status=0,
        )

    run_scaled_winding_minimize(
        capturing_minimize, fun, _X0, scale=scale, bounds=_BOUNDS
    )
    scaled_fun = captured["scaled_fun"]
    u = _X0 / scale
    analytic = scaled_fun(u)[1]
    fd = np.zeros_like(u)
    eps = 1e-6
    for i in range(len(u)):
        up = u.copy()
        up[i] += eps
        um = u.copy()
        um[i] -= eps
        fd[i] = (scaled_fun(up)[0] - scaled_fun(um)[0]) / (2.0 * eps)
    np.testing.assert_allclose(analytic, fd, rtol=1e-4)
    # The captured gradient must carry the scale factor (mutation guard: a plain
    # grad would mismatch fd above for any scaled dof).
    np.testing.assert_allclose(analytic, fun(u * scale)[1] * scale, rtol=1e-12)


def test_transform_preserves_the_bounded_minimizer():
    # Correctness: the transform is a bijection, so with enough iterations / tight
    # tolerance the scaled solve reaches the SAME physical optimum as a plain
    # solve. This proves the chain rule + scaled bounds + result inversion are
    # consistent; it is independent of whether the scale VALUE improves
    # conditioning (corridor-width scaling can under-converge soft size dofs at
    # default tolerances -- a regime-dependent tuning property, not a correctness
    # one; see WINDING_DOF_CORRIDOR_SCALE_MAP).
    fun, _ = _separable_quadratic()
    scale = build_winding_dof_scale_vector(_NAMES, WINDING_DOF_CORRIDOR_SCALE_MAP)
    opts = {"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-10}
    r_plain = minimize(
        fun, _X0, jac=True, method="L-BFGS-B", bounds=_BOUNDS, options=opts
    )
    r_scaled = run_scaled_winding_minimize(
        minimize, fun, _X0, scale=scale, bounds=_BOUNDS, options=opts
    )
    np.testing.assert_allclose(r_scaled.x, r_plain.x, atol=1e-5)
    # the inverted result is physical-space (R0 within its corridor)
    assert _BOUNDS[2][0] <= r_scaled.x[2] <= _BOUNDS[2][1]


# --- Sobolev curve-mode preconditioner (order-64 conditioning) -----------------

_SOBOLEV_NAMES = [
    "Current1:x0",                     # non-curve -> 1.0
    "SurfaceRZFourier1:rc(0,0)",       # winding size -> 1.0 (not a curve prefix)
    "CurveCWSFourierCPP1:phic(0)",     # k=0 -> 1/(1+a*0) = 1.0
    "CurveCWSFourierCPP1:phis(1)",     # k=1
    "CurveCWSFourierCPP1:thetac(8)",   # k=8
    "CurveCWSFourierCPP1:thetas(64)",  # k=64 (stiff high-order mode)
]


def test_sobolev_scale_damps_curve_modes_by_order():
    scale = build_sobolev_curve_mode_scale_vector(_SOBOLEV_NAMES, alpha=4.0, power=2)
    assert scale[0] == 1.0  # current untouched
    assert scale[1] == 1.0  # winding rc(0,0) untouched (not a curve Fourier DOF)
    assert scale[2] == 1.0  # k=0 mode: 1/(1+4*0)
    assert scale[3] == 1.0 / (1.0 + 4.0 * 1 ** 2)
    assert scale[4] == 1.0 / (1.0 + 4.0 * 8 ** 2)
    assert scale[5] == 1.0 / (1.0 + 4.0 * 64 ** 2)
    # strictly decreasing in mode order -> the higher the mode, the harder it is damped
    assert scale[3] > scale[4] > scale[5]


def test_sobolev_alpha_zero_is_identity():
    scale = build_sobolev_curve_mode_scale_vector(_SOBOLEV_NAMES, alpha=0.0)
    assert np.array_equal(scale, np.ones(len(_SOBOLEV_NAMES)))


def test_sobolev_invalid_alpha_rejected():
    import pytest

    for bad in (-1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="finite and positive"):
            build_sobolev_curve_mode_scale_vector(_SOBOLEV_NAMES, alpha=bad)


def test_sobolev_power4_more_aggressive_than_power2():
    s2 = build_sobolev_curve_mode_scale_vector(_SOBOLEV_NAMES, alpha=1.0, power=2)
    s4 = build_sobolev_curve_mode_scale_vector(_SOBOLEV_NAMES, alpha=1.0, power=4)
    assert s2[5] == 1.0 / (1.0 + 64 ** 2)
    assert s4[5] == 1.0 / (1.0 + 64 ** 4)
    assert s4[5] < s2[5]  # H^2-like damps the high mode far harder


def test_resolve_penalty_scale_off_is_winding_only_and_on_composes():
    # The solver's composer: OFF (alpha=0) must equal the winding-only scale
    # (byte-identical penalty path); ON must be the element-wise product of the
    # winding-corridor scale and the Sobolev curve-mode scale.
    from STAGE_2.banana_coil_solver import resolve_stage2_penalty_dof_scale

    winding_only = build_winding_dof_scale_vector(_NAMES, WINDING_DOF_CORRIDOR_SCALE_MAP)

    args_off = SimpleNamespace(stage2_sobolev_alpha=0.0, stage2_sobolev_power=2)
    scale_off = resolve_stage2_penalty_dof_scale(
        args_off, _NAMES, WINDING_DOF_CORRIDOR_SCALE_MAP
    )
    np.testing.assert_array_equal(scale_off, winding_only)

    args_on = SimpleNamespace(stage2_sobolev_alpha=4.0, stage2_sobolev_power=2)
    scale_on = resolve_stage2_penalty_dof_scale(
        args_on, _NAMES, WINDING_DOF_CORRIDOR_SCALE_MAP
    )
    expected = winding_only * build_sobolev_curve_mode_scale_vector(
        _NAMES, alpha=4.0, power=2
    )
    np.testing.assert_array_equal(scale_on, expected)
    # _NAMES[1] = "CurveCWSFourierCPP1:thetas(1)" -> Sobolev-damped, winding factor 1
    assert scale_on[1] == 1.0 / (1.0 + 4.0 * 1 ** 2)
    # _NAMES[2] = "SurfaceRZFourier1:rc(0,0)" -> winding-scaled, Sobolev factor 1
    assert scale_on[2] == WINDING_DOF_CORRIDOR_SCALE_MAP["rc(0,0)"]


def test_resolve_penalty_scale_composes_surface_scale():
    from STAGE_2.banana_coil_solver import resolve_stage2_penalty_dof_scale

    args = SimpleNamespace(
        stage2_sobolev_alpha=0.0,
        stage2_sobolev_power=2,
        stage2_surface_dof_scale=1.0e-4,
    )

    scale = resolve_stage2_penalty_dof_scale(
        args, _NAMES, WINDING_DOF_CORRIDOR_SCALE_MAP
    )

    expected = build_winding_dof_scale_vector(
        _NAMES, WINDING_DOF_CORRIDOR_SCALE_MAP
    ) * build_surface_dof_scale_vector(_NAMES, 1.0e-4)
    np.testing.assert_array_equal(scale, expected)
    assert scale[0] == 1.0
    assert scale[1] == 1.0
    assert scale[2] == WINDING_DOF_CORRIDOR_SCALE_MAP["rc(0,0)"] * 1.0e-4


def test_resolve_penalty_scale_composes_curve_scale():
    from STAGE_2.banana_coil_solver import resolve_stage2_penalty_dof_scale

    args = SimpleNamespace(
        stage2_sobolev_alpha=0.0,
        stage2_sobolev_power=2,
        stage2_surface_dof_scale=1.0,
        stage2_curve_dof_scale=0.25,
    )

    scale = resolve_stage2_penalty_dof_scale(
        args, _NAMES, WINDING_DOF_CORRIDOR_SCALE_MAP
    )

    expected = build_winding_dof_scale_vector(
        _NAMES, WINDING_DOF_CORRIDOR_SCALE_MAP
    ) * build_curve_dof_scale_vector(_NAMES, 0.25)
    np.testing.assert_array_equal(scale, expected)
    assert scale[0] == 1.0
    assert scale[1] == 0.25
    assert scale[2] == WINDING_DOF_CORRIDOR_SCALE_MAP["rc(0,0)"]
