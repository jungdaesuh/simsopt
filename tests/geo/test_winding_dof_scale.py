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
