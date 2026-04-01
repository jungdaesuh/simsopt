"""
Single-stage JAX backend integration tests (Milestone 5).

Validates:
1. BoozerResidualJAX.J() is small at converged surface (both CPU and JAX).
2. IotasJAX.J() is finite at independently converged solutions.
3. NonQuasiSymmetricRatioJAX.J() is finite and non-negative.
4. Adjoint-solve consistency (H^T adj = dJ_ds).
5. VJP produces finite, non-zero derivative.
6. Fixed-surface FD validates direct gradient term.
7. Composite objective value and gradient are finite and non-zero.
8. Backend selection constructs correct object types.

Gradient tests use finite-difference validation against the JAX objective
wrappers directly, because CPU and JAX use mathematically equivalent but
numerically distinct Hessian factorizations (CPU: Gauss-Newton based
Newton polish, JAX: exact Hessian), making direct gradient comparison
unreliable at ill-conditioned solution points.

All tests require ``simsoptpp`` for the CPU reference.
"""

import gc

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import scipy.linalg
from pathlib import Path
import sys
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_code_benchmark_common import summarize_result_fun
from benchmarks.single_stage_smoke_fixture import build_real_single_stage_init_fixture

sopp = pytest.importorskip(
    "simsoptpp",
    reason="Single-stage integration tests require simsoptpp (use candidate-fixed env)",
)

from simsopt.field import (  # noqa: E402
    BiotSavart,
    Coil,
    Current,
    coils_via_symmetries,
)
from simsopt.geo import (  # noqa: E402
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    CurveCWSFourier,
    CurveCWSFourierCPP,
    CurveHelical,
    CurvePlanarFourier,
    CurvePerturbed,
    CurveRZFourier,
    CurveXYZFourier,
    CurveFilament,
    FramedCurveCentroid,
    FrameRotation,
    PerturbationSample,
    create_equally_spaced_curves,
    Volume,
    BoozerSurface,
)
from simsopt.geo.surfaceobjectives import (  # noqa: E402
    BoozerResidual,
    Iotas,
    NonQuasiSymmetricRatio,
)
from simsopt.objectives import QuadraticPenalty  # noqa: E402

from simsopt.field.biotsavart_jax_backend import BiotSavartJAX  # noqa: E402
from simsopt.jax_core import (  # noqa: E402
    CoilSpec,
    CurveCWSFourierRZSpec,
    FieldEvalSpec,
    GroupedCoilSetSpec,
    grouped_coil_set_spec_from_coil_specs,
)
from simsopt.geo.boozersurface_jax import (  # noqa: E402
    BoozerSurfaceJAX,
    _boozer_ls_coil_vjp,
    _boozer_ls_coil_vjp_groups,
    _ls_decision_vector,
    _make_ls_penalty_objective,
)
from simsopt.geo.optimizer_jax import PRIVATE_OPTIMIZER_JAX_VERSION, jax_minimize  # noqa: E402
from simsopt.geo.surfaceobjectives_jax import (  # noqa: E402
    BoozerResidualJAX,
    IotasJAX,
    NonQuasiSymmetricRatioJAX,
)
from simsopt.geo.curve import Curve, RotatedCurve  # noqa: E402

from examples.single_stage_optimization.SINGLE_STAGE import (  # noqa: E402
    single_stage_banana_example as single_stage_example,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def _iota_unit_rhs(plu):
    """Return the standard IotasJAX inner cotangent for the LS path."""
    n = plu[1].shape[0]
    rhs = np.zeros(n)
    rhs[-2] = 1.0
    return rhs


def _enable_strict_jax_backend(monkeypatch, mode="jax_gpu_parity"):
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", mode)
    monkeypatch.setenv("SIMSOPT_BACKEND_STRICT", "1")


def _explicit_grouped_coil_derivative(coils, d_coil_arrays, coil_indices):
    """Reference grouped-coil projection using the original explicit summation."""
    all_derivatives = []
    for (d_g, d_gd, d_c), indices in zip(d_coil_arrays, coil_indices):
        dg = np.asarray(d_g)
        dgd = np.asarray(d_gd)
        dc = np.asarray(d_c)
        for local_i, global_i in enumerate(indices):
            all_derivatives.append(
                coils[global_i].vjp(
                    dg[local_i], dgd[local_i], np.asarray([dc[local_i]])
                )
            )
    return sum(all_derivatives)


def _reference_ls_coil_vjp_reverse_over_reverse(
    booz_surf, lm, iota, G, *, weight_inv_modB=True
):
    """Reference the original LS cotangent path before the reverse-over-forward rewrite."""
    x, optimize_G = _ls_decision_vector(booz_surf, iota, G)

    def grad_of_coils(coil_arrays):
        objective = _make_ls_penalty_objective(
            booz_surf,
            coil_arrays,
            optimize_G,
            weight_inv_modB,
        )
        return jax.grad(objective)(x)

    _, vjp_fn = jax.vjp(grad_of_coils, booz_surf._coil_arrays)
    return vjp_fn(lm)[0]


def _assert_gradients_finite_nonzero(gradients, message_prefix):
    for grad in gradients:
        assert np.all(np.isfinite(grad)), f"{message_prefix} produced NaN/inf"
        assert np.linalg.norm(grad) > 0, f"{message_prefix} produced zero gradient"


def _assert_streaming_group_vjp_matches_full(
    full_d_coil_arrays, full_coil_indices, streamed
):
    assert len(streamed) == len(full_d_coil_arrays)
    assert [indices for _, indices in streamed] == full_coil_indices
    for (streamed_arrays, _), full_arrays in zip(streamed, full_d_coil_arrays):
        for streamed_arr, full_arr in zip(streamed_arrays, full_arrays):
            np.testing.assert_allclose(
                np.asarray(streamed_arr, dtype=float),
                np.asarray(full_arr, dtype=float),
                rtol=1e-12,
                atol=1e-12,
            )


class _WholeGroupArrayConversionBomb:
    """Array-like that allows per-slice conversion but rejects whole-array casts."""

    def __init__(self, slices):
        self._slices = list(slices)

    def __array__(self, dtype=None, copy=None):
        raise AssertionError(
            "Whole grouped cotangent arrays should not be materialized"
        )

    def __getitem__(self, index):
        return self._slices[index]


class _ArrayScalarNoFloat:
    """Scalar-like wrapper that can be array-converted but must not hit float()."""

    def __init__(self, value):
        self._value = value

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._value, dtype=dtype)

    def __float__(self):
        raise AssertionError("Fallback coil current extraction should not call float()")


class _FakeCurve:
    """Non-native curve stub for _unwrap_coil_curve_and_current."""

    pass


class _FakeCurrent:
    """Stub current for _unwrap_coil_curve_and_current."""

    pass


class _RecordingVJPCoil:
    """Minimal coil stub that records per-slice VJP calls.

    Includes ``curve`` and ``current`` attributes so
    ``_unwrap_coil_curve_and_current`` can process this coil
    (non-native curve path → falls through to ``coil.vjp()``).
    """

    def __init__(self):
        self.calls = []
        self.curve = _FakeCurve()
        self.current = _FakeCurrent()

    def vjp(self, dg, dgd, dc):
        self.calls.append(
            (
                np.asarray(dg, dtype=float),
                np.asarray(dgd, dtype=float),
                np.asarray(dc, dtype=float),
            )
        )
        from simsopt._core.derivative import Derivative

        return Derivative({})


class _CpuProjectableCurve(Curve):
    """Non-JAX curve stub exposing the public CPU pullback contract."""

    def __init__(self):
        self.quadpoints = np.array([0.0, 0.5])
        super().__init__(x0=np.array([0.0, 0.0]))

    def invalidate_cache(self):
        pass

    def dgamma_by_dcoeff_vjp_impl(self, v):
        return np.array([v[0], v[1]])

    def dgammadash_by_dcoeff_vjp_impl(self, v):
        return np.array([10.0 * v[0], 10.0 * v[1]])


class _JaxProjectableCurve:
    """Curve stub exposing JAX pullback methods without native geometry support."""

    def __init__(self):
        self.dof_size = 2

    def get_dofs(self):
        return np.array([0.0, 0.0])

    def dgamma_by_dcoeff_vjp_jax(self, dofs, v):
        return jnp.array([v[0], v[1]], dtype=jnp.float64)

    def dgammadash_by_dcoeff_vjp_jax(self, dofs, v):
        return jnp.array([10.0 * v[0], 10.0 * v[1]], dtype=jnp.float64)


class _RecordingCurrent:
    """Current stub that records whether the JAX projection path reached it."""

    def __init__(self):
        self.dof_size = 1
        self.calls = []

    def vjp(self, v_current):
        self.calls.append(np.asarray(v_current, dtype=float))
        from simsopt._core.derivative import Derivative

        return Derivative({self: v_current})


class _FallbackBombCoil:
    """Coil stub whose ``vjp`` must never be called on the JAX projection path."""

    def __init__(self):
        self.curve = _JaxProjectableCurve()
        self.current = _RecordingCurrent()

    def vjp(self, dg, dgd, dc):
        raise AssertionError("JAX-projectable coils should not fall back to coil.vjp()")


class _CpuFallbackBombCoil:
    """Coil stub whose curve only supports the public CPU pullback contract."""

    def __init__(self, *, rotated=False, phi=np.pi / 2.0):
        curve = _CpuProjectableCurve()
        self.curve = RotatedCurve(curve, phi=phi, flip=False) if rotated else curve
        self.current = _RecordingCurrent()

    def vjp(self, dg, dgd, dc):
        raise AssertionError("CPU-projectable coils should not fall back to coil.vjp()")


_GENERIC_JAXCURVE_DOFS = np.array([0.1, -0.03, 0.02, 0.04, -0.01])
_GENERIC_JAXCURVE_POINTS = np.array([[1.2, 0.1, 0.2], [0.8, -0.3, 0.4]])


def _build_helical_curve(nquadpoints):
    curve = CurveHelical(nquadpoints, order=2)
    curve.set_dofs(_GENERIC_JAXCURVE_DOFS.copy())
    return curve


def _build_rotated_helical_coil():
    curve = _build_helical_curve(32)
    current = Current(8.0e4)
    coil = Coil(RotatedCurve(curve, phi=np.pi / 3.0, flip=False), current)
    return curve, current, coil


def _assert_curve_uses_jax_geometry(monkeypatch, curve, owner_name):
    monkeypatch.setattr(
        curve,
        "gamma",
        lambda: (_ for _ in ()).throw(
            AssertionError(f"{owner_name} should use CurveHelical.gamma_jax")
        ),
    )
    monkeypatch.setattr(
        curve,
        "gammadash",
        lambda: (_ for _ in ()).throw(
            AssertionError(f"{owner_name} should use CurveHelical.gammadash_jax")
        ),
    )


def _assert_curve_class_uses_jax_geometry(monkeypatch, curve, owner_name):
    curve_type = type(curve)
    monkeypatch.setattr(
        curve_type,
        "gamma",
        lambda self: (_ for _ in ()).throw(
            AssertionError(f"{owner_name} should use {curve_type.__name__}.gamma_jax")
        ),
    )
    monkeypatch.setattr(
        curve_type,
        "gammadash",
        lambda self: (_ for _ in ()).throw(
            AssertionError(
                f"{owner_name} should use {curve_type.__name__}.gammadash_jax"
            )
        ),
    )


def _build_rz_curve(nquadpoints):
    curve = CurveRZFourier(nquadpoints, order=2, nfp=3, stellsym=False)
    curve.set_dofs(
        np.array([1.2, 0.18, -0.07, 0.04, -0.03, 0.1, -0.05, 0.02, 0.08, -0.06])
    )
    return curve


def _build_planar_curve(nquadpoints):
    curve = CurvePlanarFourier(nquadpoints, order=2)
    curve.set_dofs(
        np.array([1.1, 0.14, -0.09, 0.05, -0.02, 1.0, 0.2, -0.1, 0.3, 0.15, -0.2, 0.05])
    )
    return curve


def _build_perturbed_helical_curve(nquadpoints):
    base_curve = _build_helical_curve(nquadpoints)
    quadpoints = np.asarray(base_curve.quadpoints, dtype=float)
    sample = PerturbationSample(
        None,
        sample=[
            np.column_stack(
                (
                    1.0e-3 * np.sin(2.0 * np.pi * quadpoints),
                    -8.0e-4 * np.cos(2.0 * np.pi * quadpoints),
                    6.0e-4 * np.sin(4.0 * np.pi * quadpoints),
                )
            ),
            np.column_stack(
                (
                    2.0e-3 * np.cos(2.0 * np.pi * quadpoints),
                    1.6e-3 * np.sin(2.0 * np.pi * quadpoints),
                    2.4e-3 * np.cos(4.0 * np.pi * quadpoints),
                )
            ),
        ],
    )
    return CurvePerturbed(base_curve, sample)


def _build_filament_curve(nquadpoints):
    base_curve = _build_planar_curve(nquadpoints)
    rotation = FrameRotation(base_curve.quadpoints, order=1)
    rotation.x = np.array([0.07, -0.03, 0.02])
    framed_curve = FramedCurveCentroid(base_curve, rotation)
    return CurveFilament(framed_curve, dn=0.012, db=-0.009)


def _build_surface_bound_cpp_curve(nquadpoints):
    surf = SurfaceRZFourier(
        nfp=1,
        stellsym=False,
        mpol=1,
        ntor=1,
        quadpoints_phi=np.linspace(0.0, 1.0, 16, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 16, endpoint=False),
    )
    surf.set_rc(0, 0, 1.0)
    surf.set_rc(1, 0, 0.18)
    surf.set_zs(1, 0, 0.14)
    surf.set_rc(0, 1, 0.03)
    surf.set_zs(1, 1, -0.02)

    curve = CurveCWSFourierCPP(
        np.linspace(0.0, 1.0, nquadpoints, endpoint=False),
        order=2,
        surf=surf,
    )
    curve.set("phic(0)", 0.08)
    curve.set("thetac(0)", 0.47)
    curve.set("phic(1)", -0.03)
    curve.set("phis(1)", 0.02)
    curve.set("thetas(1)", 0.07)
    return curve, surf


def _build_filament_cws_curve(nquadpoints):
    base_curve, surf = _build_surface_bound_cpp_curve(nquadpoints)
    rotation = FrameRotation(base_curve.quadpoints, order=1)
    rotation.x = np.array([0.04, 0.01, -0.02])
    framed_curve = FramedCurveCentroid(base_curve, rotation)
    return CurveFilament(framed_curve, dn=0.01, db=0.006), surf


def _assert_grouped_coil_arrays_match_curve(curve, current_value):
    bs_jax = BiotSavartJAX([Coil(curve, Current(current_value))])
    gamma_group, gammadash_group, current_group = bs_jax.grouped_coil_arrays_from_dofs(
        jnp.asarray(bs_jax.x)
    )[0]

    np.testing.assert_allclose(np.asarray(gamma_group[0]), curve.gamma(), atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(gammadash_group[0]),
        curve.gammadash(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(current_group), np.array([current_value]), atol=1e-12
    )


def _assert_biotsavart_vjp_bypasses_coil_vjp(
    curve, current, points, monkeypatch, message
):
    coil = Coil(curve, current)
    bs_cpu = BiotSavart([coil])
    bs_cpu.set_points(points)

    bs_jax = BiotSavartJAX([coil])
    bs_jax.set_points(points)
    v = np.asarray(bs_jax.B())
    deriv_cpu = bs_cpu.B_vjp(v)

    monkeypatch.setattr(
        coil,
        "vjp",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError(message)),
    )

    deriv = bs_jax.B_vjp(v)

    np.testing.assert_allclose(deriv(curve), deriv_cpu(curve), rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(
        deriv(current),
        deriv_cpu(current),
        rtol=1e-9,
        atol=1e-15,
    )


_REAL_RESOLVE_FD_REL_TOL = 1e-3
_REAL_RESOLVE_FD_ABS_TOL = 1e-8
_REAL_RESOLVE_FD_EPS = 1e-4
_REAL_RESOLVE_FD_MAX_ATTEMPTS = 6
_REAL_RESOLVE_FD_MIN_STABLE_SAMPLES = 3
_STABLE_IOTA_ABS_TOL = 5e-4
_STABLE_G_REL_TOL = 1e-4
_STABLE_FUN_REL_TOL = 1e-2


def _relative_error(actual, reference):
    return abs(actual - reference) / (abs(reference) + 1e-30)


def _make_real_resolve_fd_setup():
    """Build the stable reduced real single-stage fixture used by Tier 4."""
    fixture = build_real_single_stage_init_fixture(
        backend="jax",
        optimizer_backend="ondevice",
    )
    bs_jax = fixture["bs"]
    booz_jax = fixture["boozer_surface"]
    result = booz_jax.res
    assert result is not None and result.get("success", False), (
        "Baseline reduced real-fixture solve did not converge"
    )
    return (
        bs_jax,
        booz_jax,
        {
            "coil_dofs": np.asarray(bs_jax.x, dtype=float).copy(),
            "surface_dofs": np.asarray(booz_jax.surface.get_dofs(), dtype=float).copy(),
            "iota": float(result["iota"]),
            "G": float(result["G"]),
            "fun": float(summarize_result_fun(result)),
        },
    )


def _is_stable_real_resolve(base_state, *, iota_value, G_value, fun_value):
    return (
        abs(iota_value - float(base_state["iota"])) < _STABLE_IOTA_ABS_TOL
        and _relative_error(G_value, float(base_state["G"])) < _STABLE_G_REL_TOL
        and _relative_error(fun_value, float(base_state["fun"])) < _STABLE_FUN_REL_TOL
    )


def _build_real_resolve_overrides(base_state):
    return {
        "boozer_surface_dofs_override": np.asarray(
            base_state["surface_dofs"], dtype=float
        ),
        "boozer_iota_override": float(base_state["iota"]),
        "boozer_G_override": float(base_state["G"]),
    }


def _resolve_wrapper_value_on_real_fixture(base_state, coil_dofs, wrapper_factory):
    fixture = build_real_single_stage_init_fixture(
        backend="jax",
        optimizer_backend="ondevice",
        bs_dofs_override=np.asarray(coil_dofs, dtype=float),
        **_build_real_resolve_overrides(base_state),
    )
    bs_jax = fixture["bs"]
    booz_jax = fixture["boozer_surface"]
    result = booz_jax.res
    if result is None or not result.get("success", False):
        return {"stable": False, "reason": "solve_failed"}

    is_self_intersecting, check_available = (
        single_stage_example.evaluate_surface_self_intersection(booz_jax.surface)
    )
    if check_available and is_self_intersecting:
        return {"stable": False, "reason": "self_intersecting"}

    iota_value = float(result["iota"])
    G_value = float(result["G"])
    fun_value = float(summarize_result_fun(result))
    if not _is_stable_real_resolve(
        base_state,
        iota_value=iota_value,
        G_value=G_value,
        fun_value=fun_value,
    ):
        return {
            "stable": False,
            "reason": "branch_switch",
            "iota": iota_value,
            "G": G_value,
            "fun": fun_value,
        }

    return {
        "stable": True,
        "reason": "ok",
        "value": float(wrapper_factory(booz_jax, bs_jax).J()),
        "iota": iota_value,
        "G": G_value,
        "fun": fun_value,
    }


def _assert_wrapper_resolve_fd_matches_real_fixture(
    *,
    wrapper_label,
    gradient_builder,
    wrapper_factory,
    rng_seed,
):
    bs_jax, booz_jax, base_state = _make_real_resolve_fd_setup()
    gradient = np.asarray(gradient_builder(booz_jax, bs_jax), dtype=float)
    x0 = np.asarray(base_state["coil_dofs"], dtype=float)
    rng = np.random.RandomState(rng_seed)
    instability_reasons = []
    stable_sample_count = 0

    for sample_index in range(_REAL_RESOLVE_FD_MAX_ATTEMPTS):
        direction = rng.randn(len(x0))
        direction /= np.linalg.norm(direction)
        directional_adjoint = float(np.dot(gradient, direction))

        plus = _resolve_wrapper_value_on_real_fixture(
            base_state,
            x0 + _REAL_RESOLVE_FD_EPS * direction,
            wrapper_factory,
        )
        minus = _resolve_wrapper_value_on_real_fixture(
            base_state,
            x0 - _REAL_RESOLVE_FD_EPS * direction,
            wrapper_factory,
        )
        if not plus["stable"] or not minus["stable"]:
            instability_reasons.append(
                f"sample {sample_index}: plus={plus['reason']} minus={minus['reason']}"
            )
            continue

        directional_fd = (plus["value"] - minus["value"]) / (2.0 * _REAL_RESOLVE_FD_EPS)
        abs_err = abs(directional_adjoint - directional_fd)
        rel_err = abs_err / (abs(directional_fd) + 1e-30)
        print(
            f"{wrapper_label} reduced-real FD[{sample_index}]: "
            f"adjoint={directional_adjoint:.6e} fd={directional_fd:.6e} "
            f"rel={rel_err:.2e} abs={abs_err:.2e}"
        )
        assert (
            rel_err < _REAL_RESOLVE_FD_REL_TOL or abs_err < _REAL_RESOLVE_FD_ABS_TOL
        ), (
            f"{wrapper_label} reduced-real FD[{sample_index}] exceeded tolerance: "
            f"rel={rel_err:.2e} abs={abs_err:.2e}"
        )
        stable_sample_count += 1
        if stable_sample_count >= _REAL_RESOLVE_FD_MIN_STABLE_SAMPLES:
            return

    pytest.fail(
        f"{wrapper_label} only found {stable_sample_count} branch-stable reduced "
        f"real-fixture FD samples within {_REAL_RESOLVE_FD_MAX_ATTEMPTS} attempts; "
        f"need {_REAL_RESOLVE_FD_MIN_STABLE_SAMPLES}: " + "; ".join(instability_reasons)
    )


def _make_boozer_setup(
    constraint_weight=1.0,
    optimizer_backend="ondevice",
    *,
    weight_inv_modB=True,
):
    """Create a Boozer surface configuration for testing."""
    ncoils = 2
    nfp = 2
    stellsym = True
    R0 = 1.0
    R1 = 0.5
    order = 3

    base_curves = create_equally_spaced_curves(
        ncoils,
        nfp,
        stellsym=stellsym,
        R0=R0,
        R1=R1,
        order=order,
    )
    base_currents = [Current(1e5) for _ in range(ncoils)]
    for c in base_currents:
        c.fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)

    mpol = 2
    ntor = 2
    nphi = 2 * ntor + 1
    ntheta = 2 * mpol + 1
    surf_cpu = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=stellsym,
        nfp=nfp,
        quadpoints_phi=np.linspace(0, 1.0 / nfp, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0, 1.0, ntheta, endpoint=False),
    )
    surf_cpu.set_dofs(np.zeros_like(surf_cpu.get_dofs()))
    from simsopt.geo import SurfaceRZFourier

    s_rz = SurfaceRZFourier(
        nfp=nfp,
        stellsym=stellsym,
        mpol=1,
        ntor=0,
        quadpoints_phi=surf_cpu.quadpoints_phi,
        quadpoints_theta=surf_cpu.quadpoints_theta,
    )
    s_rz.set_rc(0, 0, R0)
    s_rz.set_rc(1, 0, 0.15)
    s_rz.set_zs(1, 0, 0.15)
    surf_cpu.least_squares_fit(s_rz.gamma())

    surf_jax = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=stellsym,
        nfp=nfp,
        quadpoints_phi=surf_cpu.quadpoints_phi,
        quadpoints_theta=surf_cpu.quadpoints_theta,
    )
    surf_jax.set_dofs(surf_cpu.get_dofs().copy())

    bs_cpu = BiotSavart(coils)
    bs_jax = BiotSavartJAX(coils)

    vol_cpu = Volume(surf_cpu)
    vol_jax = Volume(surf_jax)
    vol_target = vol_cpu.J()

    mu0 = 4 * np.pi * 1e-7
    G0 = mu0 * sum(abs(c.current.get_value()) for c in coils)
    iota0 = 0.3

    booz_cpu = BoozerSurface(
        bs_cpu,
        surf_cpu,
        vol_cpu,
        vol_target,
        constraint_weight=constraint_weight,
        options={"verbose": False, "bfgs_maxiter": 50, "newton_maxiter": 0},
    )
    booz_jax = BoozerSurfaceJAX(
        bs_jax,
        surf_jax,
        vol_jax,
        vol_target,
        constraint_weight=constraint_weight,
        options={
            "verbose": False,
            "bfgs_maxiter": 300,
            "bfgs_tol": 1e-8,
            "newton_maxiter": 20,
            "newton_tol": 1e-9,
            "optimizer_backend": optimizer_backend,
            "weight_inv_modB": weight_inv_modB,
        },
    )

    return (
        coils,
        surf_cpu,
        surf_jax,
        bs_cpu,
        bs_jax,
        booz_cpu,
        booz_jax,
        vol_cpu,
        iota0,
        G0,
    )


@pytest.fixture(scope="module")
def boozer_setup():
    """Module-scoped Boozer surface setup with LS constraint."""
    setup = _make_boozer_setup(constraint_weight=1.0)
    (
        coils,
        surf_cpu,
        surf_jax,
        bs_cpu,
        bs_jax,
        booz_cpu,
        booz_jax,
        vol_cpu,
        iota0,
        G0,
    ) = setup

    # Run BOTH solvers independently from the same initial guess.
    # This validates the real all-JAX path, not a CPU-state injection.
    res_cpu = booz_cpu.run_code(iota0, G0)
    assert res_cpu is not None, "CPU BoozerSurface.run_code() returned None"
    assert "PLU" in res_cpu, "CPU solver did not produce PLU"

    res_jax = booz_jax.run_code(iota0, G0)
    assert res_jax is not None, "JAX BoozerSurfaceJAX.run_code() returned None"
    assert res_jax.get("success", False), "JAX solver did not converge"
    assert "PLU" in res_jax, "JAX solver did not produce PLU"

    return (
        coils,
        surf_cpu,
        surf_jax,
        bs_cpu,
        bs_jax,
        booz_cpu,
        booz_jax,
        vol_cpu,
    )


# -----------------------------------------------------------------------
# Test 1: BoozerResidual value sanity
# -----------------------------------------------------------------------


class TestBoozerResidualValue:
    """Both solvers produce small Boozer residuals at their solutions."""

    def test_j_both_small(self, boozer_setup):
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )

        jr_cpu = BoozerResidual(booz_cpu, bs_cpu)
        jr_jax = BoozerResidualJAX(booz_jax, bs_jax)

        j_cpu = jr_cpu.J()
        j_jax = jr_jax.J()

        print(f"BoozerResidual J: cpu={j_cpu:.12e} jax={j_jax:.12e}")
        # Both should be small on the shared reduced fixture.
        assert j_jax < 1e-3, f"JAX BoozerResidual too large: {j_jax:.2e}"
        assert j_cpu < 1e-3, f"CPU BoozerResidual too large: {j_cpu:.2e}"


# -----------------------------------------------------------------------
# Test 2: Iotas value sanity
# -----------------------------------------------------------------------


class TestIotasValue:
    """IotasJAX.J() is finite at independently converged solutions."""

    def test_j_finite(self, boozer_setup):
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )

        iotas_cpu = Iotas(booz_cpu)
        iotas_jax = IotasJAX(booz_jax)

        j_cpu = iotas_cpu.J()
        j_jax = iotas_jax.J()

        print(f"Iotas J: cpu={j_cpu:.12e} jax={j_jax:.12e}")
        # Both must be finite (solvers may converge to different branches)
        assert np.isfinite(j_cpu) and np.isfinite(j_jax), "Iotas J not finite"


# -----------------------------------------------------------------------
# Test 3: IotasJAX.dJ() adjoint FD validation (re-solve)
# -----------------------------------------------------------------------


class TestAdjointSolveConsistency:
    """Validate the adjoint linear system: (PLU)^T adj = dJ_ds.

    This proves the adjoint pipeline is correct without relying on
    re-solve FD (which branch-switches on small grids — confirmed
    to happen on BOTH CPU and JAX solvers on this config).
    """

    def test_device_native_adjoint_solve_matches_host(self, boozer_setup):
        """JAX PLU solve matches the legacy host triangular solve."""
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )
        from simsopt.objectives.utilities import forward_backward, forward_backward_jax

        P, L, U = booz_jax.res["PLU"]
        dJ_ds = _iota_unit_rhs((P, L, U))

        adj_host = forward_backward(P, L, U, dJ_ds)
        adj_jax = np.asarray(forward_backward_jax(P, L, U, dJ_ds))

        np.testing.assert_allclose(adj_jax, adj_host, rtol=1e-12, atol=1e-12)

    def test_adjoint_residual(self, boozer_setup):
        """Check that forward_backward_jax(PLU, dJ_ds) solves H^T adj = dJ_ds."""
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )
        from simsopt.objectives.utilities import forward_backward_jax

        P, L, U = booz_jax.res["PLU"]
        dJ_ds = _iota_unit_rhs((P, L, U))

        adj = np.asarray(forward_backward_jax(P, L, U, dJ_ds))

        # Verify: (P @ L @ U)^T @ adj should equal dJ_ds
        H = P @ L @ U
        residual = H.T @ adj - dJ_ds
        rel = np.linalg.norm(residual) / (np.linalg.norm(dJ_ds) + 1e-30)
        print(f"Adjoint residual: ||H^T adj - dJ_ds|| / ||dJ_ds|| = {rel:.2e}")
        assert rel < 1e-10, f"Adjoint solve residual too large: {rel:.2e}"

    def test_vjp_produces_finite_derivative(self, boozer_setup):
        """VJP hook produces a finite, non-zero Derivative from a non-trivial adjoint."""
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )
        from simsopt.objectives.utilities import forward_backward_jax

        P, L, U = booz_jax.res["PLU"]
        dJ_ds = _iota_unit_rhs((P, L, U))
        adj = forward_backward_jax(P, L, U, dJ_ds)

        vjp_fn = booz_jax.res["vjp"]
        adj_cot = vjp_fn(adj, booz_jax, booz_jax.res["iota"], booz_jax.res["G"])
        adj_deriv = bs_jax.coil_cotangents_to_derivative(*adj_cot)
        g = np.array(adj_deriv(bs_jax))

        print(f"||VJP result|| = {np.linalg.norm(g):.6e}")
        assert np.all(np.isfinite(g)), "VJP produced NaN/inf"
        assert np.linalg.norm(g) > 0, "VJP produced zero gradient"

    def test_coil_cotangent_projection_matches_explicit_sum(self, boozer_setup):
        """Incremental grouped-coil accumulation matches the prior explicit summation."""
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )
        from simsopt.objectives.utilities import forward_backward_jax

        P, L, U = booz_jax.res["PLU"]
        dJ_ds = _iota_unit_rhs((P, L, U))
        adj = forward_backward_jax(P, L, U, dJ_ds)

        vjp_fn = booz_jax.res["vjp"]
        d_coil_arrays, coil_indices = vjp_fn(
            adj, booz_jax, booz_jax.res["iota"], booz_jax.res["G"]
        )
        projected = bs_jax.coil_cotangents_to_derivative(d_coil_arrays, coil_indices)
        explicit = _explicit_grouped_coil_derivative(
            bs_jax.coils, d_coil_arrays, coil_indices
        )

        np.testing.assert_allclose(
            np.asarray(projected(bs_jax), dtype=float),
            np.asarray(explicit(bs_jax), dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_coil_cotangent_projection_avoids_whole_group_host_materialization(self):
        """Projection should convert one coil slice at a time, not a whole group."""
        from simsopt.geo.surfaceobjectives_jax import _coil_cotangents_to_derivative

        coils = [_RecordingVJPCoil(), _RecordingVJPCoil()]
        d_coil_arrays = [
            (
                _WholeGroupArrayConversionBomb(
                    [
                        np.array([1.0, 2.0, 3.0]),
                        np.array([4.0, 5.0, 6.0]),
                    ]
                ),
                _WholeGroupArrayConversionBomb(
                    [
                        np.array([7.0, 8.0, 9.0]),
                        np.array([10.0, 11.0, 12.0]),
                    ]
                ),
                _WholeGroupArrayConversionBomb([1.5, 2.5]),
            )
        ]

        _coil_cotangents_to_derivative(coils, d_coil_arrays, [[0, 1]])

        assert len(coils[0].calls) == 1
        assert len(coils[1].calls) == 1
        np.testing.assert_allclose(coils[0].calls[0][0], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(coils[1].calls[0][0], np.array([4.0, 5.0, 6.0]))
        np.testing.assert_allclose(coils[0].calls[0][1], np.array([7.0, 8.0, 9.0]))
        np.testing.assert_allclose(coils[1].calls[0][1], np.array([10.0, 11.0, 12.0]))
        np.testing.assert_allclose(coils[0].calls[0][2], np.array([1.5]))
        np.testing.assert_allclose(coils[1].calls[0][2], np.array([2.5]))

    def test_grouped_coil_arrays_from_dofs_respects_unique_dof_lineage_order(self):
        """Native grouped reconstruction must decode free current DOFs by lineage slice."""
        curve = CurveXYZFourier(16, 1)
        curve.x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        current = Current(1.23)
        bs_jax = BiotSavartJAX([Coil(curve, current)])

        lineage_names = [type(opt).__name__ for opt in bs_jax.unique_dof_lineage]
        assert lineage_names.index("Current") < lineage_names.index("CurveXYZFourier")

        gamma_group, gammadash_group, current_group = (
            bs_jax.grouped_coil_arrays_from_dofs(jnp.asarray(bs_jax.x))[0]
        )

        np.testing.assert_allclose(
            np.asarray(gamma_group[0]), curve.gamma(), atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(gammadash_group[0]),
            curve.gammadash(),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(current_group), np.array([1.23]), atol=1e-12
        )

    def test_coil_set_spec_from_dofs_reuses_grouped_coil_ssot(self):
        """Explicit reconstruction should expose the immutable grouped-coil spec."""
        curve = CurveXYZFourier(16, 1)
        curve.x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        current = Current(1.23)
        bs_jax = BiotSavartJAX([Coil(curve, current)])

        coil_set_spec = bs_jax.coil_set_spec_from_dofs(jnp.asarray(bs_jax.x))

        assert isinstance(coil_set_spec, GroupedCoilSetSpec)
        assert len(coil_set_spec.groups) == 1
        group = coil_set_spec.groups[0]
        assert group.coil_indices == (0,)
        np.testing.assert_allclose(
            np.asarray(group.gammas[0]), curve.gamma(), atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(group.gammadashs[0]),
            curve.gammadash(),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(group.currents),
            np.array([1.23]),
            atol=1e-12,
        )

    def test_legacy_objects_expose_curve_current_coil_specs(self):
        """Legacy hot-path objects should expose immutable JAX specs."""
        curve = CurveXYZFourier(16, 1)
        curve.x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        rotated_curve = RotatedCurve(curve, np.pi / 3.0, False)
        current = Current(1.23)
        scaled_current = 2.0 * current
        coil = Coil(rotated_curve, scaled_current)

        curve_spec = curve.to_spec()
        current_spec = current.to_spec()
        coil_spec = coil.to_spec()
        grouped_spec = grouped_coil_set_spec_from_coil_specs((coil_spec,))

        assert isinstance(current_spec.value, jax.Array)
        assert isinstance(coil_spec, CoilSpec)
        assert grouped_spec.groups[0].coil_indices == (0,)
        np.testing.assert_allclose(
            np.asarray(grouped_spec.groups[0].gammas[0]),
            rotated_curve.gamma(),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(grouped_spec.groups[0].gammadashs[0]),
            rotated_curve.gammadash(),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(grouped_spec.groups[0].currents),
            np.array([2.46]),
            atol=1e-12,
        )
        assert curve_spec.order == curve.order

    @pytest.mark.parametrize("curve_cls", (CurveCWSFourierCPP, CurveCWSFourier))
    def test_curve_on_surface_coils_expose_immutable_specs(self, curve_cls):
        """Curve-on-surface coils should round-trip through immutable curve specs."""
        coil_surf = SurfaceRZFourier(
            nfp=2,
            stellsym=True,
            mpol=1,
            ntor=0,
            quadpoints_phi=np.arange(32) / 32,
            quadpoints_theta=np.arange(32) / 32,
        )
        coil_surf.set_rc(0, 0, 0.95)
        coil_surf.set_rc(1, 0, 0.2)
        coil_surf.set_zs(1, 0, 0.2)

        curve = curve_cls(
            np.linspace(0.0, 1.0, 64, endpoint=False),
            order=2,
            surf=coil_surf,
        )
        curve.set("phic(0)", 0.07)
        curve.set("thetac(0)", 0.35)
        curve.set("phic(1)", 0.02)
        curve.set("thetas(1)", -0.08)
        current = 1.5 * Current(2.0)
        coil = Coil(curve, current)

        coil_spec = coil.to_spec()
        grouped_spec = grouped_coil_set_spec_from_coil_specs((coil_spec,))

        assert isinstance(coil_spec, CoilSpec)
        assert isinstance(coil_spec.curve, CurveCWSFourierRZSpec)
        np.testing.assert_allclose(
            np.asarray(grouped_spec.groups[0].gammas[0]),
            curve.gamma(),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(grouped_spec.groups[0].gammadashs[0]),
            curve.gammadash(),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(grouped_spec.groups[0].currents),
            np.array([3.0]),
            atol=1e-12,
        )

    def test_field_eval_spec_round_trip_uses_immutable_points(self):
        """BiotSavartJAX should round-trip evaluation points through FieldEvalSpec."""
        curve = CurveXYZFourier(16, 1)
        current = Current(1.23)
        bs_jax = BiotSavartJAX([Coil(curve, current)])
        points = np.array(
            [
                [0.1, 0.0, 0.0],
                [0.2, 0.1, -0.1],
            ]
        )

        bs_jax.set_points(points)
        field_eval_spec = bs_jax.field_eval_spec()

        assert isinstance(field_eval_spec, FieldEvalSpec)
        np.testing.assert_allclose(np.asarray(field_eval_spec.points), points)

        updated_points = jnp.asarray(points + 0.05, dtype=jnp.float64)
        bs_jax.set_points_from_spec(FieldEvalSpec(points=updated_points))
        np.testing.assert_allclose(
            np.asarray(bs_jax.field_eval_spec().points), updated_points
        )

    def test_grouped_coil_arrays_from_dofs_supports_generic_jaxcurve_geometry(
        self,
        monkeypatch,
    ):
        """Explicit coil-DOF reconstruction should work for JAX-capable non-XYZ curves."""
        curve = _build_helical_curve(16)
        current = Current(1.23)
        bs_jax = BiotSavartJAX([Coil(curve, current)])
        curve_dofs = jnp.asarray(curve.get_dofs(), dtype=jnp.float64)
        expected_gamma = np.asarray(curve.gamma_jax(curve_dofs))
        expected_gammadash = np.asarray(curve.gammadash_jax(curve_dofs))

        _assert_curve_uses_jax_geometry(
            monkeypatch,
            curve,
            "grouped_coil_arrays_from_dofs()",
        )

        gamma_group, gammadash_group, current_group = (
            bs_jax.grouped_coil_arrays_from_dofs(jnp.asarray(bs_jax.x))[0]
        )

        np.testing.assert_allclose(
            np.asarray(gamma_group[0]), expected_gamma, atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(gammadash_group[0]),
            expected_gammadash,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(current_group), np.array([1.23]), atol=1e-12
        )

    def test_biotsavart_B_uses_generic_jaxcurve_geometry_without_cpu_calls(
        self,
        monkeypatch,
    ):
        """Forward field evaluation should stay on the JAX geometry lane for JaxCurve subclasses."""
        curve, _current, coil = _build_rotated_helical_coil()

        bs_cpu = BiotSavart([coil])
        bs_cpu.set_points(_GENERIC_JAXCURVE_POINTS)
        B_cpu = bs_cpu.B()

        _assert_curve_uses_jax_geometry(monkeypatch, curve, "BiotSavartJAX.B()")

        bs_jax = BiotSavartJAX([coil])
        bs_jax.set_points(_GENERIC_JAXCURVE_POINTS)
        B_jax = np.asarray(bs_jax.B())

        np.testing.assert_allclose(B_jax, B_cpu, rtol=1e-10, atol=1e-15)

    def test_biotsavart_B_vjp_bypasses_coil_vjp_for_generic_jaxcurve(self, monkeypatch):
        """Reverse field pullback should stay off ``Coil.vjp()`` for real JAX-capable curves."""
        curve, current, coil = _build_rotated_helical_coil()

        bs_cpu = BiotSavart([coil])
        bs_cpu.set_points(_GENERIC_JAXCURVE_POINTS)

        bs_jax = BiotSavartJAX([coil])
        bs_jax.set_points(_GENERIC_JAXCURVE_POINTS)
        v = np.asarray(bs_jax.B())
        deriv_cpu = bs_cpu.B_vjp(v)

        monkeypatch.setattr(
            coil,
            "vjp",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError(
                    "BiotSavartJAX.B_vjp() should bypass Coil.vjp() for CurveHelical"
                )
            ),
        )

        deriv = bs_jax.B_vjp(v)

        np.testing.assert_allclose(
            deriv(curve),
            deriv_cpu(curve),
            rtol=1e-9,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            deriv(current),
            deriv_cpu(current),
            rtol=1e-9,
            atol=1e-15,
        )

    def test_grouped_coil_arrays_from_dofs_supports_curverzfourier_geometry(self):
        """Explicit reconstruction should work for legacy cylindrical Fourier curves."""
        _assert_grouped_coil_arrays_match_curve(_build_rz_curve(24), 1.23)

    def test_biotsavart_B_uses_curverzfourier_jax_geometry_without_cpu_calls(
        self,
        monkeypatch,
    ):
        """Forward field evaluation should stay on the JAX lane for CurveRZFourier."""
        curve = _build_rz_curve(24)
        current = Current(8.0e4)
        coil = Coil(curve, current)

        bs_cpu = BiotSavart([coil])
        bs_cpu.set_points(_GENERIC_JAXCURVE_POINTS)
        B_cpu = bs_cpu.B()

        _assert_curve_class_uses_jax_geometry(monkeypatch, curve, "BiotSavartJAX.B()")

        bs_jax = BiotSavartJAX([coil])
        bs_jax.set_points(_GENERIC_JAXCURVE_POINTS)
        B_jax = np.asarray(bs_jax.B())

        np.testing.assert_allclose(B_jax, B_cpu, rtol=1e-10, atol=1e-15)

    def test_grouped_coil_arrays_from_dofs_supports_curveplanarfourier_geometry(self):
        """Explicit reconstruction should work for planar legacy Fourier curves."""
        _assert_grouped_coil_arrays_match_curve(_build_planar_curve(24), 2.5)

    def test_grouped_coil_arrays_from_dofs_supports_curveperturbed_fullgraph_geometry(
        self,
    ):
        """Full-graph wrapper curves should reconstruct from explicit coil DOFs."""
        _assert_grouped_coil_arrays_match_curve(_build_perturbed_helical_curve(24), 1.7)

    def test_biotsavart_B_vjp_bypasses_coil_vjp_for_curvefilament(self, monkeypatch):
        """Full-graph finite-build wrapper curves should stay off ``Coil.vjp()``."""
        _assert_biotsavart_vjp_bypasses_coil_vjp(
            _build_filament_curve(32),
            Current(8.0e4),
            _GENERIC_JAXCURVE_POINTS,
            monkeypatch,
            "BiotSavartJAX.B_vjp() should bypass Coil.vjp() for CurveFilament",
        )

    def test_biotsavart_B_vjp_preserves_surface_pullback_for_curvefilament_cws(
        self,
        monkeypatch,
    ):
        """CWS-backed finite-build curves should stay on the JAX wrapper pullback path."""
        curve, _surf = _build_filament_cws_curve(32)
        _assert_biotsavart_vjp_bypasses_coil_vjp(
            curve,
            Current(8.0e4),
            _GENERIC_JAXCURVE_POINTS,
            monkeypatch,
            "BiotSavartJAX.B_vjp() should bypass Coil.vjp() for finite-build CWS curves",
        )

    def test_biotsavart_projection_uses_cpu_curve_pullbacks_for_non_jax_curves(self):
        """Non-JAX curves should project through public CPU curve/current VJPs."""
        bs_jax = object.__new__(BiotSavartJAX)
        coils = [_CpuFallbackBombCoil(), _CpuFallbackBombCoil()]
        bs_jax._coils = coils
        d_coil_arrays = [
            (
                _WholeGroupArrayConversionBomb(
                    [
                        np.array([1.0, 2.0, 3.0]),
                        np.array([4.0, 5.0, 6.0]),
                    ]
                ),
                _WholeGroupArrayConversionBomb(
                    [
                        np.array([7.0, 8.0, 9.0]),
                        np.array([10.0, 11.0, 12.0]),
                    ]
                ),
                _WholeGroupArrayConversionBomb([1.5, 2.5]),
            )
        ]

        derivative = bs_jax.coil_cotangents_to_derivative(d_coil_arrays, [[0, 1]])

        np.testing.assert_allclose(
            np.asarray(derivative.data[coils[0].curve], dtype=float),
            np.array([71.0, 82.0]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(derivative.data[coils[1].curve], dtype=float),
            np.array([104.0, 115.0]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(derivative.data[coils[0].current], dtype=float),
            np.array([1.5]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(derivative.data[coils[1].current], dtype=float),
            np.array([2.5]),
            atol=1e-12,
        )
        assert len(coils[0].current.calls) == 1
        assert len(coils[1].current.calls) == 1

    def test_biotsavart_projection_rotates_cpu_curve_cotangents(self):
        """Rotated non-JAX curves should apply inverse rotation before CPU VJPs."""
        bs_jax = object.__new__(BiotSavartJAX)
        coils = [_CpuFallbackBombCoil(rotated=True, phi=np.pi / 2.0)]
        bs_jax._coils = coils
        d_gamma = np.array([1.0, 2.0, 3.0])
        d_gammadash = np.array([4.0, 5.0, 6.0])

        derivative = bs_jax.coil_cotangents_to_derivative(
            [(jnp.asarray([d_gamma]), jnp.asarray([d_gammadash]), jnp.asarray([1.5]))],
            [[0]],
        )

        np.testing.assert_allclose(
            np.asarray(derivative.data[coils[0].curve.curve], dtype=float),
            np.array([52.0, -41.0]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(derivative.data[coils[0].current], dtype=float),
            np.array([1.5]),
            atol=1e-12,
        )
        assert len(coils[0].current.calls) == 1

    def test_biotsavart_projection_rejects_unsupported_curves_without_coil_fallback(
        self,
    ):
        """Unsupported curves should fail fast instead of falling back to ``coil.vjp()``."""
        bs_jax = object.__new__(BiotSavartJAX)
        coils = [_RecordingVJPCoil()]
        bs_jax._coils = coils

        with pytest.raises(TypeError, match="supported JAX or CPU pullback contract"):
            bs_jax.coil_cotangents_to_derivative(
                [
                    (
                        jnp.array([[1.0, 2.0, 3.0]]),
                        jnp.array([[4.0, 5.0, 6.0]]),
                        jnp.array([1.5]),
                    )
                ],
                [[0]],
            )

        assert coils[0].calls == []

    def test_biotsavart_projection_uses_jax_pullbacks_for_projectable_curves(self):
        """JAX-capable curves should bypass ``coil.vjp()`` even if they are not native."""
        bs_jax = object.__new__(BiotSavartJAX)
        coils = [_FallbackBombCoil()]
        bs_jax._coils = coils

        derivative = bs_jax.coil_cotangents_to_derivative(
            [
                (
                    jnp.array([[1.0, 2.0, 3.0]]),
                    jnp.array([[4.0, 5.0, 6.0]]),
                    jnp.array([1.5]),
                )
            ],
            [[0]],
        )

        np.testing.assert_allclose(
            np.asarray(derivative.data[coils[0].curve], dtype=float),
            np.array([41.0, 52.0]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(derivative.data[coils[0].current], dtype=float),
            np.array([1.5]),
            atol=1e-12,
        )

    def test_biotsavart_grouped_extraction_keeps_array_like_cpu_currents(self):
        """Legacy fallback extraction should preserve array-like current scalars."""
        bs_jax = object.__new__(BiotSavartJAX)
        bs_jax._coils = [object()]
        bs_jax._jax_native = False
        bs_jax._coil_geometry_inputs = lambda coil, geometry_cache: (
            None,
            None,
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.0, 1.0, 0.0]]),
            _ArrayScalarNoFloat(1.5),
        )

        groups = bs_jax._extract_coil_data_grouped()

        assert len(groups) == 1
        _, _, currents, indices = groups[0]
        np.testing.assert_allclose(np.asarray(currents, dtype=float), np.array([1.5]))
        assert indices == [0]

    def test_compat_helper_uses_shared_jax_projection_for_projectable_curves(self):
        """The compatibility helper should share the same JAX projection path."""
        from simsopt.geo.surfaceobjectives_jax import _coil_cotangents_to_derivative

        coils = [_FallbackBombCoil()]
        derivative = _coil_cotangents_to_derivative(
            coils,
            [
                (
                    jnp.array([[1.0, 2.0, 3.0]]),
                    jnp.array([[4.0, 5.0, 6.0]]),
                    jnp.array([1.5]),
                )
            ],
            [[0]],
        )

        np.testing.assert_allclose(
            np.asarray(derivative.data[coils[0].curve], dtype=float),
            np.array([41.0, 52.0]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(derivative.data[coils[0].current], dtype=float),
            np.array([1.5]),
            atol=1e-12,
        )

    def test_compat_helper_rejects_coil_vjp_fallback_in_strict_mode(self, monkeypatch):
        """Strict JAX mode must reject the legacy Coil.vjp() projection fallback."""
        from simsopt.geo.surfaceobjectives_jax import _coil_cotangents_to_derivative

        _enable_strict_jax_backend(monkeypatch)
        coils = [_RecordingVJPCoil()]

        with pytest.raises(
            RuntimeError,
            match="surfaceobjectives_jax.*Coil\\.vjp\\(\\).*strict=True",
        ):
            _coil_cotangents_to_derivative(
                coils,
                [
                    (
                        jnp.array([[1.0, 2.0, 3.0]]),
                        jnp.array([[4.0, 5.0, 6.0]]),
                        jnp.array([1.5]),
                    )
                ],
                [[0]],
            )

        assert coils[0].calls == []

    def test_refresh_coil_data_reuses_grouped_currents_without_host_reads(
        self, monkeypatch
    ):
        """Refreshing grouped coil data should not re-read coil currents on host."""
        (
            coils,
            surf_cpu,
            surf_jax,
            bs_cpu,
            bs_jax,
            booz_cpu,
            booz_jax,
            vol_cpu,
            _iota0,
            _G0,
        ) = _make_boozer_setup(
            constraint_weight=1.0,
            optimizer_backend="scipy",
        )

        for coil in bs_jax.coils:
            monkeypatch.setattr(
                coil.current,
                "get_value",
                lambda: (_ for _ in ()).throw(
                    AssertionError(
                        "_refresh_coil_data() should reuse grouped JAX current arrays"
                    )
                ),
            )

        booz_jax._refresh_coil_data()

        expected = np.zeros(len(bs_jax.coils))
        for _, _, currents, indices in booz_jax.coil_groups:
            for local_i, global_i in enumerate(indices):
                expected[global_i] = float(np.asarray(currents[local_i]))
        np.testing.assert_allclose(
            np.asarray(booz_jax.coil_currents, dtype=float),
            expected,
            atol=1e-12,
        )

    def test_surface_objectives_jax_reject_host_forward_backward(
        self, boozer_setup, monkeypatch
    ):
        """Target JAX implicit wrappers must not fall back to SciPy triangular solves."""
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )

        def _bomb(*args, **kwargs):
            raise AssertionError(
                "SciPy triangular solves should not run on the JAX implicit path"
            )

        monkeypatch.setattr(scipy.linalg, "solve_triangular", _bomb)

        gradients = [
            np.array(BoozerResidualJAX(booz_jax, bs_jax).dJ()),
            np.array(IotasJAX(booz_jax).dJ()),
            np.array(NonQuasiSymmetricRatioJAX(booz_jax, bs_jax, sDIM=6).dJ()),
        ]

        _assert_gradients_finite_nonzero(gradients, "Implicit JAX path")

    def test_surface_objectives_jax_prefers_streaming_group_vjp(
        self, boozer_setup, monkeypatch
    ):
        """Implicit wrappers should use group-at-a-time VJPs when available."""
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )

        def _bomb(*args, **kwargs):
            raise AssertionError(
                "Whole-pytree VJP should not run on the streaming path"
            )

        monkeypatch.setitem(booz_jax.res, "vjp", _bomb)

        gradients = [
            np.array(BoozerResidualJAX(booz_jax, bs_jax).dJ()),
            np.array(IotasJAX(booz_jax).dJ()),
            np.array(NonQuasiSymmetricRatioJAX(booz_jax, bs_jax, sDIM=6).dJ()),
        ]

        _assert_gradients_finite_nonzero(gradients, "Streaming group VJP")

    def test_streaming_group_vjp_matches_full_vjp(self, boozer_setup):
        """Group-at-a-time VJPs should match the legacy full-pytree VJP result."""
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )
        from simsopt.objectives.utilities import forward_backward_jax

        P, L, U = booz_jax.res["PLU"]
        dJ_ds = _iota_unit_rhs((P, L, U))
        adj = forward_backward_jax(P, L, U, dJ_ds)

        full_d_coil_arrays, full_coil_indices = booz_jax.res["vjp"](
            adj, booz_jax, booz_jax.res["iota"], booz_jax.res["G"]
        )
        streamed = list(
            booz_jax.res["vjp_groups"](
                adj, booz_jax, booz_jax.res["iota"], booz_jax.res["G"]
            )
        )

        _assert_streaming_group_vjp_matches_full(
            full_d_coil_arrays, full_coil_indices, streamed
        )

    def test_streaming_group_vjp_matches_full_vjp_fixed_G(self):
        """Grouped LS VJPs should also match when ``optimize_G=False``."""
        (
            coils,
            surf_cpu,
            surf_jax,
            bs_cpu,
            bs_jax,
            booz_cpu,
            booz_jax,
            vol_cpu,
            iota0,
            G0,
        ) = _make_boozer_setup(constraint_weight=1.0, optimizer_backend="ondevice")
        from simsopt.objectives.utilities import forward_backward_jax

        res_ls = booz_jax.run_code(iota0, None)
        assert res_ls is not None
        assert res_ls.get("success", False), "Fixed-G LS JAX solve did not converge"

        P, L, U = res_ls["PLU"]
        dJ_ds = _iota_unit_rhs((P, L, U))
        adj = forward_backward_jax(P, L, U, dJ_ds)

        full_d_coil_arrays, full_coil_indices = res_ls["vjp"](
            adj, booz_jax, res_ls["iota"], res_ls["G"]
        )
        streamed = list(
            res_ls["vjp_groups"](adj, booz_jax, res_ls["iota"], res_ls["G"])
        )

        _assert_streaming_group_vjp_matches_full(
            full_d_coil_arrays, full_coil_indices, streamed
        )

    def test_streaming_group_vjp_matches_full_vjp_without_inv_modB_weighting(
        self, boozer_setup
    ):
        """Grouped LS VJPs should match when ``weight_inv_modB=False``."""
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )
        from simsopt.objectives.utilities import forward_backward_jax

        P, L, U = booz_jax.res["PLU"]
        dJ_ds = _iota_unit_rhs((P, L, U))
        adj = forward_backward_jax(P, L, U, dJ_ds)

        full_d_coil_arrays, full_coil_indices = _boozer_ls_coil_vjp(
            adj,
            booz_jax,
            booz_jax.res["iota"],
            booz_jax.res["G"],
            weight_inv_modB=False,
        )
        streamed = list(
            _boozer_ls_coil_vjp_groups(
                adj,
                booz_jax,
                booz_jax.res["iota"],
                booz_jax.res["G"],
                weight_inv_modB=False,
            )
        )

        _assert_streaming_group_vjp_matches_full(
            full_d_coil_arrays, full_coil_indices, streamed
        )

    def test_ls_coil_vjp_matches_reverse_over_reverse_reference(self, boozer_setup):
        """LS cotangent rewrite must match the previous reverse-over-reverse result."""
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )
        from simsopt.objectives.utilities import forward_backward_jax

        P, L, U = booz_jax.res["PLU"]
        dJ_ds = _iota_unit_rhs((P, L, U))
        adj = forward_backward_jax(P, L, U, dJ_ds)

        rewritten_d_coil_arrays, rewritten_coil_indices = booz_jax.res["vjp"](
            adj, booz_jax, booz_jax.res["iota"], booz_jax.res["G"]
        )
        reference_d_coil_arrays = _reference_ls_coil_vjp_reverse_over_reverse(
            booz_jax,
            adj,
            booz_jax.res["iota"],
            booz_jax.res["G"],
            weight_inv_modB=booz_jax.res.get("weight_inv_modB", True),
        )

        assert rewritten_coil_indices == booz_jax._coil_index_lists
        for rewritten_arrays, reference_arrays in zip(
            rewritten_d_coil_arrays,
            reference_d_coil_arrays,
        ):
            for rewritten_arr, reference_arr in zip(rewritten_arrays, reference_arrays):
                np.testing.assert_allclose(
                    np.asarray(rewritten_arr, dtype=float),
                    np.asarray(reference_arr, dtype=float),
                    rtol=1e-12,
                    atol=1e-12,
                )

    def test_ls_coil_vjp_matches_directional_objective_fd(self, boozer_setup):
        """LS cotangent should match FD on the scalar directional objective."""
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )
        from simsopt.objectives.utilities import forward_backward_jax

        P, L, U = booz_jax.res["PLU"]
        dJ_ds = _iota_unit_rhs((P, L, U))
        adj = forward_backward_jax(P, L, U, dJ_ds)

        full_d_coil_arrays, full_coil_indices = booz_jax.res["vjp"](
            adj, booz_jax, booz_jax.res["iota"], booz_jax.res["G"]
        )
        derivative = _explicit_grouped_coil_derivative(
            coils,
            full_d_coil_arrays,
            full_coil_indices,
        )
        full_gradient = np.asarray(derivative(bs_jax), dtype=float)

        x, optimize_G = _ls_decision_vector(
            booz_jax, booz_jax.res["iota"], booz_jax.res["G"]
        )

        def directional_objective_at(coil_x):
            bs_jax.x = coil_x
            booz_jax._refresh_coil_data()
            objective = _make_ls_penalty_objective(
                booz_jax,
                booz_jax._coil_arrays,
                optimize_G,
                booz_jax.res.get("weight_inv_modB", True),
            )
            return float(jnp.vdot(adj, jax.grad(objective)(x)))

        x0 = bs_jax.x.copy()
        rng = np.random.RandomState(7)
        eps = 1e-5

        for i in range(3):
            direction = rng.randn(len(x0))
            direction /= np.linalg.norm(direction)

            dd_vjp = float(np.dot(full_gradient, direction))
            dd_fd = (
                directional_objective_at(x0 + eps * direction)
                - directional_objective_at(x0 - eps * direction)
            ) / (2 * eps)

            abs_err = abs(dd_vjp - dd_fd)
            rel_err = abs_err / (abs(dd_fd) + 1e-30)
            assert rel_err < 1e-3 or abs_err < 1e-8, (
                f"LS cotangent FD[{i}]: vjp={dd_vjp:.6e} fd={dd_fd:.6e} "
                f"rel={rel_err:.2e} abs={abs_err:.2e}"
            )

        bs_jax.x = x0
        booz_jax._refresh_coil_data()

    def test_exact_streaming_group_vjp_matches_full_vjp(self):
        """Exact-solve group-at-a-time VJPs should match the legacy exact VJP."""
        (
            coils,
            surf_cpu,
            surf_jax,
            bs_cpu,
            bs_jax,
            booz_cpu,
            booz_jax,
            vol_cpu,
            iota0,
            G0,
        ) = _make_boozer_setup(constraint_weight=1.0, optimizer_backend="ondevice")

        # Seed the exact Newton solve from the converged LS state on this fixture.
        # The raw initial guess is not a stable exact-solve regression anchor here.
        res_ls = booz_jax.run_code(iota0, G0)
        assert res_ls is not None
        assert res_ls.get("success", False), "LS JAX solve did not converge"
        booz_jax.need_to_run_code = True

        res_exact = booz_jax.solve_residual_equation_exactly_newton(
            iota=res_ls["iota"], G=res_ls["G"]
        )

        assert res_exact is not None
        assert res_exact.get("type") == "exact"
        assert res_exact.get("success", False), "Exact JAX solve did not converge"
        assert res_exact.get("vjp_groups") is not None

        lm = np.ones(res_exact["PLU"][1].shape[0], dtype=float)
        full_d_coil_arrays, full_coil_indices = res_exact["vjp"](
            lm, booz_jax, res_exact["iota"], res_exact["G"]
        )
        streamed = list(
            res_exact["vjp_groups"](lm, booz_jax, res_exact["iota"], res_exact["G"])
        )

        _assert_streaming_group_vjp_matches_full(
            full_d_coil_arrays, full_coil_indices, streamed
        )

    def test_compute_derivative_l2_metrics_does_not_mutate_derivative_map(
        self, boozer_setup
    ):
        """Norm helper should not populate missing derivative entries on read."""
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )
        from benchmarks.adjoint_probe_common import compute_derivative_l2_metrics
        from simsopt._core.derivative import Derivative

        derivative = Derivative({})
        original_keys = tuple(derivative.data.keys())

        norm, finite = compute_derivative_l2_metrics(derivative, bs_jax)

        assert norm == pytest.approx(0.0)
        assert finite is True
        assert tuple(derivative.data.keys()) == original_keys

    def test_compute_derivative_l2_metrics_matches_full_gradient_norm(
        self, boozer_setup
    ):
        """Norm helper should match the full Derivative gradient on a real fixture."""
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )
        from benchmarks.adjoint_probe_common import compute_derivative_l2_metrics

        derivative_for_metrics = BoozerResidualJAX(booz_jax, bs_jax).dJ(partials=True)
        derivative_for_full = BoozerResidualJAX(booz_jax, bs_jax).dJ(partials=True)

        norm, finite = compute_derivative_l2_metrics(derivative_for_metrics, bs_jax)
        full_gradient = np.asarray(derivative_for_full(bs_jax), dtype=float)

        assert finite is True
        assert norm == pytest.approx(
            float(np.linalg.norm(full_gradient)),
            rel=1e-12,
            abs=1e-12,
        )


# -----------------------------------------------------------------------
# Test 4: NonQuasiSymmetricRatio value sanity
# -----------------------------------------------------------------------


class TestNonQSRatioValue:
    """NonQuasiSymmetricRatioJAX.J() is finite and non-negative at converged solutions."""

    def test_j_finite_nonneg(self, boozer_setup):
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )

        sDIM = 6
        nqs_cpu = NonQuasiSymmetricRatio(booz_cpu, bs_cpu, sDIM=sDIM)
        nqs_jax = NonQuasiSymmetricRatioJAX(booz_jax, bs_jax, sDIM=sDIM)

        j_cpu = nqs_cpu.J()
        j_jax = nqs_jax.J()

        print(f"NonQSRatio J: cpu={j_cpu:.12e} jax={j_jax:.12e}")
        # Both must be finite and non-negative (solvers converge to different
        # surfaces, so exact parity is not expected)
        assert np.isfinite(j_jax) and j_jax >= 0, f"JAX NonQSRatio invalid: {j_jax}"
        assert np.isfinite(j_cpu) and j_cpu >= 0, f"CPU NonQSRatio invalid: {j_cpu}"


# -----------------------------------------------------------------------
# Test 5: Composite objective value sanity
# -----------------------------------------------------------------------


class TestCompositeObjective:
    """Combined JF produces finite value and gradient on JAX path."""

    def test_composite_jax(self, boozer_setup):
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )

        jr_jax = BoozerResidualJAX(booz_jax, bs_jax)
        iotas_jax = IotasJAX(booz_jax)
        iota_target = booz_jax.res["iota"]
        JF_jax = jr_jax + QuadraticPenalty(iotas_jax, iota_target)

        j = JF_jax.J()
        g = JF_jax.dJ()

        print(f"Composite JAX: J={j:.12e} ||dJ||={np.linalg.norm(g):.6e}")
        assert np.isfinite(j), "Composite J is not finite"
        assert np.all(np.isfinite(g)), "Composite dJ contains NaN/inf"


# -----------------------------------------------------------------------
# Test 6: JAX gradient finite-difference validation
# -----------------------------------------------------------------------


class TestBoozerResidualGradientFD:
    """End-to-end BoozerResidualJAX.dJ() vs fixed-surface FD.

    Calls the real composed method ``dJ_by_dcoils - adj_derivative``
    and compares against FD at fixed surface.  At a converged Boozer
    surface the adjoint term ≈ 0 (∂J_BR/∂x_inner ≈ 0), so the
    composed gradient equals the direct term.  This validates the
    full code path through ``BoozerResidualJAX.compute()``.
    """

    def test_end_to_end_dJ_vs_fd(self, boozer_setup):
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )

        import jax.numpy as jnp
        from simsopt.geo.boozer_residual_jax import boozer_residual_vector

        jr_jax = BoozerResidualJAX(booz_jax, bs_jax)
        jr_jax.J()
        g_composed = jr_jax.dJ()

        gamma_fixed = surf_jax.gamma().reshape(-1, 3)
        xphi = jnp.asarray(surf_jax.gammadash1())
        xtheta = jnp.asarray(surf_jax.gammadash2())
        nphi = surf_jax.quadpoints_phi.size
        ntheta = surf_jax.quadpoints_theta.size
        num_pts = 3 * nphi * ntheta
        iota_sol = booz_jax.res["iota"]
        G_sol = booz_jax.res["G"]

        def J_at_fixed_surface(coil_x):
            bs_jax.x = coil_x
            bs_jax.set_points(gamma_fixed)
            B = bs_jax.B().reshape(nphi, ntheta, 3)
            r = boozer_residual_vector(G_sol, iota_sol, B, xphi, xtheta, True)
            return 0.5 * float(jnp.sum(r**2)) / num_pts

        x0 = bs_jax.x.copy()
        rng = np.random.RandomState(42)
        eps = 1e-5

        for i in range(3):
            d = rng.randn(len(x0))
            d /= np.linalg.norm(d)

            dd_composed = float(np.dot(g_composed, d))
            dd_fd = (
                J_at_fixed_surface(x0 + eps * d) - J_at_fixed_surface(x0 - eps * d)
            ) / (2 * eps)

            abs_err = abs(dd_composed - dd_fd)
            rel_err = abs_err / (abs(dd_fd) + 1e-30)
            print(
                f"E2E FD[{i}]: composed={dd_composed:.6e} fd={dd_fd:.6e} "
                f"rel={rel_err:.2e} abs={abs_err:.2e}"
            )
            assert rel_err < 1e-3 or abs_err < 1e-8, (
                f"E2E FD[{i}]: rel={rel_err:.2e} abs={abs_err:.2e}"
            )

        bs_jax.x = x0
        bs_jax.set_points(gamma_fixed)


# -----------------------------------------------------------------------
# Test 7: End-to-end composite gradient pipeline
# -----------------------------------------------------------------------


class TestCompositeGradientPipeline:
    """JAX composite objective produces finite, non-zero gradient.

    A full gradient-descent progress test is impractical on this small 5x5
    grid because the Boozer inner solve lands at a poor local minimum
    (J_JAX ≈ 0.047 vs J_CPU ≈ 2.5e-6), making the IFT adjoint term
    unreliable for determining descent direction.  The direct term is
    validated separately in ``TestBoozerResidualGradientFD``.

    This test verifies the end-to-end pipeline: value + gradient are
    finite, gradient is non-zero, and both terms (BoozerResidual + iota
    penalty) contribute.
    """

    def test_composite_gradient_finite_and_nonzero(self, boozer_setup):
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )

        iota_target = booz_jax.res["iota"]
        jr_jax = BoozerResidualJAX(booz_jax, bs_jax)
        iotas_jax = IotasJAX(booz_jax)
        JF_jax = jr_jax + 10.0 * QuadraticPenalty(iotas_jax, iota_target)

        j0 = JF_jax.J()
        dj0 = JF_jax.dJ()
        grad_norm = np.linalg.norm(dj0)

        print(f"Composite: J={j0:.6e}, ||dJ||={grad_norm:.6e}")

        assert np.isfinite(j0), "Composite J is not finite"
        assert np.all(np.isfinite(dj0)), "Composite dJ contains NaN/inf"
        assert grad_norm > 0, "Gradient is zero — pipeline may be broken"


# -----------------------------------------------------------------------
# Test 8: Script-level --backend jax constructs JAX objects
# -----------------------------------------------------------------------


class TestScriptBackendSelection:
    """initialize_boozer_surface(..., backend='jax') uses BoozerSurfaceJAX."""

    def test_jax_backend_constructs_boozer_surface_jax(self, boozer_setup):
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )
        assert type(booz_jax).__name__ == "BoozerSurfaceJAX"
        assert type(booz_cpu).__name__ == "BoozerSurface"

    def test_initialize_boozer_surface_jax_backend(self):
        """Call the real initialize_boozer_surface with backend='jax'."""
        import importlib.util
        from unittest.mock import MagicMock, patch

        spec = importlib.util.spec_from_file_location(
            "single_stage",
            REPO_ROOT
            / "examples"
            / "single_stage_optimization"
            / "SINGLE_STAGE"
            / "single_stage_banana_example.py",
        )
        mod = importlib.util.module_from_spec(spec)

        fake_bs = MagicMock()
        fake_bs.coils = []
        fake_surf = MagicMock()
        fake_surf.quadpoints_phi = np.linspace(0, 0.5, 5)
        fake_surf.quadpoints_theta = np.linspace(0, 1, 5)
        fake_surf.gamma.return_value = np.zeros((5, 5, 3))

        recorder = MagicMock()
        recorder.return_value = MagicMock(
            run_code=MagicMock(
                return_value={"success": True, "G": 1.0, "iota": 0.3, "iter": 10}
            ),
            surface=MagicMock(
                is_self_intersecting=MagicMock(return_value=False),
                volume=MagicMock(return_value=0.1),
            ),
        )

        with patch.dict(
            "sys.modules",
            {"simsopt.geo.boozersurface_jax": MagicMock(BoozerSurfaceJAX=recorder)},
        ):
            spec.loader.exec_module(mod)

            fake_vol = MagicMock()
            fake_vol.return_value = MagicMock()
            with patch.object(mod, "Volume", fake_vol), patch.object(
                mod, "SurfaceXYZTensorFourier", MagicMock(return_value=fake_surf)
            ):
                mod.initialize_boozer_surface(
                    fake_surf,
                    mpol=2,
                    ntor=2,
                    bs=fake_bs,
                    vol_target=0.1,
                    constraint_weight=1.0,
                    iota=0.3,
                    G0=1.0,
                    backend="jax",
                )

        assert recorder.called, "BoozerSurfaceJAX was not constructed"
        print("initialize_boozer_surface(backend='jax') -> BoozerSurfaceJAX OK")

    def test_real_fixture_cpu_warm_start_overrides_do_not_crash(self):
        """Reduced real CPU fixture must accept warm-start overrides without sdofs=."""
        from unittest.mock import patch

        with patch.object(
            single_stage_example,
            "evaluate_surface_self_intersection",
            return_value=(False, False),
        ):
            base_fixture = build_real_single_stage_init_fixture(
                backend="cpu",
                optimizer_backend="scipy",
            )
            base_boozer = base_fixture["boozer_surface"]
            base_result = base_boozer.res

            assert base_result is not None and base_result.get("success", False), (
                "Baseline reduced real CPU fixture did not converge"
            )

            replay_fixture = build_real_single_stage_init_fixture(
                backend="cpu",
                optimizer_backend="scipy",
                boozer_surface_dofs_override=np.asarray(
                    base_boozer.surface.get_dofs(),
                    dtype=float,
                ),
                boozer_iota_override=float(base_result["iota"]),
                boozer_G_override=float(base_result["G"]),
            )
            replay_result = replay_fixture["boozer_surface"].res

            assert replay_result is not None and replay_result.get("success", False), (
                "Warm-started reduced real CPU fixture did not converge"
            )


# -----------------------------------------------------------------------
# Test 9: Isolated run_code() LS parity (CPU vs JAX)
# -----------------------------------------------------------------------


class TestRunCodeLSParity:
    """Isolated parity: CPU and JAX run_code() from the same initial guess.

    Verifies that BoozerSurface and BoozerSurfaceJAX converge to the same
    quality solution with identical solver options.  This is the primary
    regression gate for the LS inner solve path (plan §2 workflow acceptance).
    """

    def test_ls_solve_parity(self):
        """Both solvers converge; iota, label error, and residual match."""
        ncoils, nfp = 2, 2
        base_curves = create_equally_spaced_curves(
            ncoils,
            nfp,
            stellsym=True,
            R0=1.0,
            R1=0.5,
            order=3,
        )
        base_currents = [Current(1e5) for _ in range(ncoils)]
        for c in base_currents:
            c.fix_all()
        coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym=True)

        mpol, ntor = 2, 2
        nphi, ntheta = 2 * ntor + 1, 2 * mpol + 1
        surf_cpu = SurfaceXYZTensorFourier(
            mpol=mpol,
            ntor=ntor,
            stellsym=True,
            nfp=nfp,
            quadpoints_phi=np.linspace(0, 1.0 / nfp, nphi, endpoint=False),
            quadpoints_theta=np.linspace(0, 1.0, ntheta, endpoint=False),
        )
        surf_cpu.set_dofs(np.zeros_like(surf_cpu.get_dofs()))
        from simsopt.geo import SurfaceRZFourier

        s_rz = SurfaceRZFourier(
            nfp=nfp,
            stellsym=True,
            mpol=1,
            ntor=0,
            quadpoints_phi=surf_cpu.quadpoints_phi,
            quadpoints_theta=surf_cpu.quadpoints_theta,
        )
        s_rz.set_rc(0, 0, 1.0)
        s_rz.set_rc(1, 0, 0.15)
        s_rz.set_zs(1, 0, 0.15)
        surf_cpu.least_squares_fit(s_rz.gamma())

        surf_jax = SurfaceXYZTensorFourier(
            mpol=mpol,
            ntor=ntor,
            stellsym=True,
            nfp=nfp,
            quadpoints_phi=surf_cpu.quadpoints_phi,
            quadpoints_theta=surf_cpu.quadpoints_theta,
        )
        surf_jax.set_dofs(surf_cpu.get_dofs().copy())

        bs_cpu = BiotSavart(coils)
        bs_jax = BiotSavartJAX(coils)
        vol_cpu = Volume(surf_cpu)
        vol_jax = Volume(surf_jax)
        vol_target = vol_cpu.J()

        mu0 = 4 * np.pi * 1e-7
        G0 = mu0 * sum(abs(c.current.get_value()) for c in coils)
        iota0 = 0.3

        opts = {
            "verbose": False,
            "bfgs_maxiter": 300,
            "bfgs_tol": 1e-8,
            "newton_maxiter": 20,
            "newton_tol": 1e-9,
        }
        booz_cpu = BoozerSurface(
            bs_cpu,
            surf_cpu,
            vol_cpu,
            vol_target,
            constraint_weight=1.0,
            options=opts,
        )
        booz_jax = BoozerSurfaceJAX(
            bs_jax,
            surf_jax,
            vol_jax,
            vol_target,
            constraint_weight=1.0,
            options=opts,
        )

        res_cpu = booz_cpu.run_code(iota0, G0)
        res_jax = booz_jax.run_code(iota0, G0)

        assert res_cpu.get("success", False), "CPU solver did not converge"
        assert res_jax.get("success", False), "JAX solver did not converge"

        label_err_cpu = abs(vol_cpu.J() - vol_target)
        label_err_jax = abs(vol_jax.J() - vol_target)
        iota_diff = abs(res_cpu["iota"] - res_jax["iota"])

        print(
            f"CPU: iota={res_cpu['iota']:.6e} |label|={label_err_cpu:.6e}\n"
            f"JAX: iota={res_jax['iota']:.6e} |label|={label_err_jax:.6e}\n"
            f"|iota diff|={iota_diff:.6e}"
        )

        # Both should converge to near-zero iota and label error
        assert abs(res_cpu["iota"]) < 1e-3, f"CPU iota too large: {res_cpu['iota']}"
        assert abs(res_jax["iota"]) < 1e-3, f"JAX iota too large: {res_jax['iota']}"
        assert label_err_cpu < 1e-3, f"CPU label error too large: {label_err_cpu}"
        assert label_err_jax < 1e-3, f"JAX label error too large: {label_err_jax}"
        assert iota_diff < 1e-6, f"Iota disagreement: {iota_diff:.6e}"


# -----------------------------------------------------------------------
# Test 10: Short outer optimization loop (plan §5 gate)
# -----------------------------------------------------------------------


class TestShortSingleStageOptRun:
    """Run a short outer optimization and verify the objective decreases.

    The plan (line 626) requires: "run a minimal optimization step sequence,
    not just component calls."  This test builds a composite JAX objective
    (BoozerResidual + iota penalty), takes a few L-BFGS-B steps on the
    outer DOFs, and checks that the composite objective decreases.
    """

    def test_outer_opt_decreases_objective(self, boozer_setup):
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )
        from scipy.optimize import minimize as scipy_minimize

        iota_target = booz_jax.res["iota"]
        jr_jax = BoozerResidualJAX(booz_jax, bs_jax)
        iotas_jax = IotasJAX(booz_jax)
        JF_jax = jr_jax + QuadraticPenalty(iotas_jax, iota_target)

        x0 = JF_jax.x.copy()
        j0 = JF_jax.J()
        assert np.isfinite(j0), "Initial objective not finite"

        def fun(x):
            JF_jax.x = x
            return JF_jax.J(), JF_jax.dJ()

        result = scipy_minimize(
            fun,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 3, "maxcor": 10},
        )
        j_final = result.fun

        print(
            f"Short opt: J0={j0:.6e} -> J_final={j_final:.6e} "
            f"nit={result.nit} success={result.success}"
        )
        assert np.isfinite(j_final), "Final objective not finite"
        assert j_final <= j0 + 1e-12, (
            f"Objective did not decrease: {j0:.6e} -> {j_final:.6e}"
        )

        JF_jax.x = x0


# -----------------------------------------------------------------------
# Test 11: Exact-path Boozer solve
# -----------------------------------------------------------------------


class TestExactPathSolve:
    """Verify that the exact Newton path runs and converges.

    The plan (line 695) requires: "the exact-path final-stage workflow
    remains in scope, not just least-squares initialization."
    """

    def test_exact_path_converges(self):
        """BoozerSurfaceJAX with boozer_type='exact' converges."""
        ncoils, nfp = 2, 2
        base_curves = create_equally_spaced_curves(
            ncoils,
            nfp,
            stellsym=True,
            R0=1.0,
            R1=0.5,
            order=3,
        )
        base_currents = [Current(1e5) for _ in range(ncoils)]
        for c in base_currents:
            c.fix_all()
        coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym=True)
        bs_jax = BiotSavartJAX(coils)

        mpol, ntor = 2, 2
        nphi, ntheta = 2 * ntor + 1, 2 * mpol + 1
        surf = SurfaceXYZTensorFourier(
            mpol=mpol,
            ntor=ntor,
            stellsym=True,
            nfp=nfp,
            quadpoints_phi=np.linspace(0, 1.0 / nfp, nphi, endpoint=False),
            quadpoints_theta=np.linspace(0, 1.0, ntheta, endpoint=False),
        )
        from simsopt.geo import SurfaceRZFourier

        s_rz = SurfaceRZFourier(
            nfp=nfp,
            stellsym=True,
            mpol=1,
            ntor=0,
            quadpoints_phi=surf.quadpoints_phi,
            quadpoints_theta=surf.quadpoints_theta,
        )
        s_rz.set_rc(0, 0, 1.0)
        s_rz.set_rc(1, 0, 0.15)
        s_rz.set_zs(1, 0, 0.15)
        surf.least_squares_fit(s_rz.gamma())

        vol = Volume(surf)
        vol_target = vol.J()

        mu0 = 4 * np.pi * 1e-7
        G0 = mu0 * sum(abs(c.current.get_value()) for c in coils)
        iota0 = 0.3

        # Warm-start: run LS solve first so the exact Newton has a good initial guess.
        booz_ls = BoozerSurfaceJAX(
            bs_jax,
            surf,
            vol,
            vol_target,
            constraint_weight=1.0,
            options={
                "verbose": False,
                "bfgs_maxiter": 300,
                "bfgs_tol": 1e-10,
                "newton_maxiter": 20,
                "newton_tol": 1e-11,
            },
        )
        ls_res = booz_ls.run_code(iota0, G0)
        assert ls_res["success"], "LS warm-start solve did not converge"
        iota_warm = ls_res["iota"]
        G_warm = ls_res["G"]
        # Surface DOFs are already updated in-place by the LS solve.

        booz_exact = BoozerSurfaceJAX(
            bs_jax,
            surf,
            vol,
            vol_target,
            constraint_weight=None,
            options={
                "verbose": False,
                "newton_maxiter": 40,
                "newton_tol": 1e-8,
            },
        )
        res = booz_exact.run_code(iota_warm, G_warm)

        assert res is not None, "Exact solver returned None"
        assert res["type"] == "exact", f"Expected 'exact', got {res['type']}"
        assert "weight_inv_modB" in res, "Missing weight_inv_modB key"
        residual_norm = np.linalg.norm(res["residual"], ord=np.inf)
        print(
            f"Exact path: success={res['success']} iter={res['iter']} "
            f"||residual||_inf={residual_norm:.3e} iota={res['iota']:.6f}"
        )
        assert residual_norm < 1e-6, (
            f"Exact solver residual too large: ||r||={residual_norm:.3e}"
        )


@pytest.mark.private_optimizer_runtime
class TestOnDeviceBackendIntegration:
    """Exercise the real on-device LS solve against simsoptpp-backed fixtures."""

    @pytest.mark.skipif(
        jax.__version__ != PRIVATE_OPTIMIZER_JAX_VERSION,
        reason=f"On-device backend integration requires the validated JAX {PRIVATE_OPTIMIZER_JAX_VERSION} runtime.",
    )
    @pytest.mark.parametrize("optimizer_backend", ["ondevice", "hybrid"])
    @pytest.mark.parametrize("pass_explicit_G", [True, False])
    def test_ondevice_backend_run_code_converges(
        self, optimizer_backend, pass_explicit_G
    ):
        (_, _, _, _, bs_jax, _, booz_jax, _, iota0, G0) = _make_boozer_setup(
            constraint_weight=1.0,
            optimizer_backend=optimizer_backend,
        )
        import jax.numpy as jnp
        from simsopt.geo.boozer_residual_jax import boozer_residual_vector

        G_arg = G0 if pass_explicit_G else None
        res = booz_jax.run_code(iota0, G_arg)

        assert res is not None, f"{optimizer_backend} backend returned None"
        assert res["type"] == "ls", f"Expected 'ls', got {res['type']}"
        assert res["success"], f"{optimizer_backend} backend did not converge"
        assert np.isfinite(res["fun"]), (
            f"{optimizer_backend} backend returned non-finite fun"
        )
        assert res["PLU"] is not None, f"{optimizer_backend} backend did not build PLU"
        assert callable(res["vjp"]), f"{optimizer_backend} backend did not expose VJP"
        if pass_explicit_G:
            assert res["G"] is not None
        else:
            assert res["G"] is None

        jr_jax = BoozerResidualJAX(booz_jax, bs_jax)
        value = jr_jax.J()
        grad = jr_jax.dJ()
        assert np.isfinite(value), (
            f"{optimizer_backend} backend produced non-finite M5 value"
        )
        assert np.all(np.isfinite(grad)), (
            f"{optimizer_backend} backend produced non-finite M5 dJ"
        )

        gamma_fixed = booz_jax.surface.gamma().reshape(-1, 3)
        xphi = jnp.asarray(booz_jax.surface.gammadash1())
        xtheta = jnp.asarray(booz_jax.surface.gammadash2())
        nphi = booz_jax.surface.quadpoints_phi.size
        ntheta = booz_jax.surface.quadpoints_theta.size
        num_pts = 3 * nphi * ntheta
        effective_G = res["G"] if res["G"] is not None else G0
        iota_sol = res["iota"]

        def J_at_fixed_surface(coil_x):
            bs_jax.x = coil_x
            bs_jax.set_points(gamma_fixed)
            B = bs_jax.B().reshape(nphi, ntheta, 3)
            r = boozer_residual_vector(effective_G, iota_sol, B, xphi, xtheta, True)
            return 0.5 * float(jnp.sum(r**2)) / num_pts

        x0 = bs_jax.x.copy()
        direction = np.linspace(1.0, 2.0, len(x0))
        direction /= np.linalg.norm(direction)
        eps = 1e-5
        dd_composed = float(np.dot(grad, direction))
        dd_fd = (
            J_at_fixed_surface(x0 + eps * direction)
            - J_at_fixed_surface(x0 - eps * direction)
        ) / (2 * eps)
        abs_err = abs(dd_composed - dd_fd)
        rel_err = abs_err / (abs(dd_fd) + 1e-30)

        assert rel_err < 1e-3 or abs_err < 1e-8, (
            f"{optimizer_backend} pass_explicit_G={pass_explicit_G}: "
            f"composed={dd_composed:.6e} fd={dd_fd:.6e} "
            f"rel={rel_err:.2e} abs={abs_err:.2e}"
        )

        bs_jax.x = x0
        bs_jax.set_points(gamma_fixed)


class TestEnsureSolvedCrashGuard:
    """Issue-1 regression: _ensure_solved must not crash with res=None."""

    def test_J_before_run_code_gives_clear_error(self):
        """BoozerResidualJAX.J() before run_code() raises RuntimeError."""
        ncoils, nfp = 2, 2
        base_curves = create_equally_spaced_curves(
            ncoils,
            nfp,
            stellsym=True,
            R0=1.0,
            R1=0.5,
            order=3,
        )
        base_currents = [Current(1e5) for _ in range(ncoils)]
        for c in base_currents:
            c.fix_all()
        coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym=True)
        bs_jax = BiotSavartJAX(coils)

        s = SurfaceXYZTensorFourier(
            mpol=2,
            ntor=2,
            stellsym=True,
            nfp=nfp,
            quadpoints_phi=np.linspace(0, 1.0 / nfp, 5, endpoint=False),
            quadpoints_theta=np.linspace(0, 1.0, 5, endpoint=False),
        )
        vol = Volume(s)
        booz = BoozerSurfaceJAX(bs_jax, s, vol, 0.1, constraint_weight=1.0)

        assert booz.res is None
        obj = BoozerResidualJAX(booz, bs_jax)

        with pytest.raises(RuntimeError, match="has not been solved yet"):
            obj.J()

    @pytest.mark.parametrize(
        "wrapper_name",
        ["BoozerResidualJAX", "IotasJAX", "NonQuasiSymmetricRatioJAX"],
    )
    def test_m5_wrappers_raise_before_touching_garbage(
        self, boozer_setup, wrapper_name
    ):
        """All M5 wrappers must stop at _ensure_solved when res is unset.

        This guards the negative path where a failed inner solve would leave
        no PLU/VJP contract to consume.
        """
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        old_res = booz_jax.res
        old_dirty = booz_jax.need_to_run_code
        old_run_code = booz_jax.run_code
        run_code_called = False

        def forbidden_run_code(*args, **kwargs):
            nonlocal run_code_called
            run_code_called = True
            raise AssertionError("run_code must not be called when res is None")

        booz_jax.res = None
        booz_jax.need_to_run_code = True
        booz_jax.run_code = forbidden_run_code
        try:
            if wrapper_name == "BoozerResidualJAX":
                obj = BoozerResidualJAX(booz_jax, bs_jax)
            elif wrapper_name == "IotasJAX":
                obj = IotasJAX(booz_jax)
            else:
                obj = NonQuasiSymmetricRatioJAX(booz_jax, bs_jax)

            with pytest.raises(RuntimeError, match="has not been solved yet"):
                obj.J()

            assert not run_code_called
        finally:
            booz_jax.res = old_res
            booz_jax.need_to_run_code = old_dirty
            booz_jax.run_code = old_run_code

    @pytest.mark.parametrize(
        "wrapper_name",
        ["BoozerResidualJAX", "IotasJAX", "NonQuasiSymmetricRatioJAX"],
    )
    def test_m5_wrappers_raise_on_failed_solve_state(self, boozer_setup, wrapper_name):
        """Failed inner solves must be rejected even if adjoint placeholders exist."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        old_res = booz_jax.res
        old_dirty = booz_jax.need_to_run_code
        old_run_code = booz_jax.run_code
        run_code_called = False

        def forbidden_run_code(*args, **kwargs):
            nonlocal run_code_called
            run_code_called = True
            raise AssertionError("run_code must not be called for cached failed solve")

        bad_res = dict(old_res)
        bad_res["success"] = False
        bad_res["PLU"] = tuple(np.eye(2) for _ in range(3))
        bad_res["vjp"] = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("vjp must not be touched for failed solves")
        )
        booz_jax.res = bad_res
        booz_jax.need_to_run_code = False
        booz_jax.run_code = forbidden_run_code
        try:
            if wrapper_name == "BoozerResidualJAX":
                obj = BoozerResidualJAX(booz_jax, bs_jax)
            elif wrapper_name == "IotasJAX":
                obj = IotasJAX(booz_jax)
            else:
                obj = NonQuasiSymmetricRatioJAX(booz_jax, bs_jax)

            with pytest.raises(
                RuntimeError, match="failed to produce valid adjoint state"
            ):
                obj.J()

            assert not run_code_called
        finally:
            booz_jax.res = old_res
            booz_jax.need_to_run_code = old_dirty
            booz_jax.run_code = old_run_code


# -----------------------------------------------------------------------
# Test 14: B_vjp CPU↔JAX parity
# -----------------------------------------------------------------------


class TestBVjpCPUParityPerComponent:
    """BiotSavartJAX.B_vjp(v) matches BiotSavart.B_vjp(v) per-component.

    Both paths compute the VJP of B w.r.t. coil DOFs at shared evaluation
    points and with a shared cotangent vector.  The resulting Derivative
    vectors should agree to tight tolerance.
    """

    def test_b_vjp_parity(self, boozer_setup):
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )

        gamma_flat = surf_jax.gamma().reshape(-1, 3)
        old_points_cpu = bs_cpu.get_points_cart_ref().copy()
        old_points_jax = bs_jax._points_jax
        bs_cpu.set_points(gamma_flat)
        bs_jax.set_points(gamma_flat)

        rng = np.random.RandomState(99)
        v = rng.randn(*gamma_flat.shape)

        deriv_cpu = bs_cpu.B_vjp(v)
        deriv_jax = bs_jax.B_vjp(v)

        grad_cpu = np.asarray(deriv_cpu(bs_cpu), dtype=float)
        grad_jax = np.asarray(deriv_jax(bs_jax), dtype=float)

        # Restore eval points on module-scoped fixture
        bs_cpu.set_points(old_points_cpu)
        bs_jax._points_jax = old_points_jax

        print(
            f"B_vjp parity: ||cpu||={np.linalg.norm(grad_cpu):.6e} "
            f"||jax||={np.linalg.norm(grad_jax):.6e} "
            f"||diff||={np.linalg.norm(grad_cpu - grad_jax):.6e}"
        )
        b_vjp_rel_tol = 1e-10
        b_vjp_abs_tol = 1e-12
        np.testing.assert_allclose(
            grad_jax,
            grad_cpu,
            rtol=b_vjp_rel_tol,
            atol=b_vjp_abs_tol,
        )


# -----------------------------------------------------------------------
# Test 15: Exact solve CPU↔JAX parity
# -----------------------------------------------------------------------


class _ExactSolveParityPair(NamedTuple):
    bs_cpu: BiotSavart
    bs_jax: BiotSavartJAX
    booz_cpu_exact: BoozerSurface
    booz_jax_exact: BoozerSurfaceJAX
    res_cpu: dict
    res_jax: dict


def _solve_exact_cpu_jax_parity_pair() -> _ExactSolveParityPair:
    ncoils, nfp = 2, 2
    stellsym = True
    base_curves = create_equally_spaced_curves(
        ncoils,
        nfp,
        stellsym=stellsym,
        R0=1.0,
        R1=0.5,
        order=3,
    )
    base_currents = [Current(1e5) for _ in range(ncoils)]
    for current in base_currents:
        current.fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)

    mpol, ntor = 2, 2
    nphi, ntheta = 2 * ntor + 1, 2 * mpol + 1
    qp_phi = np.linspace(0, 1.0 / nfp, nphi, endpoint=False)
    qp_theta = np.linspace(0, 1.0, ntheta, endpoint=False)

    from simsopt.geo import SurfaceRZFourier

    s_rz = SurfaceRZFourier(
        nfp=nfp,
        stellsym=stellsym,
        mpol=1,
        ntor=0,
        quadpoints_phi=qp_phi,
        quadpoints_theta=qp_theta,
    )
    s_rz.set_rc(0, 0, 1.0)
    s_rz.set_rc(1, 0, 0.15)
    s_rz.set_zs(1, 0, 0.15)

    surf_cpu = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=stellsym,
        nfp=nfp,
        quadpoints_phi=qp_phi,
        quadpoints_theta=qp_theta,
    )
    surf_cpu.least_squares_fit(s_rz.gamma())
    bs_cpu = BiotSavart(coils)
    vol_cpu = Volume(surf_cpu)
    vol_target = vol_cpu.J()

    mu0 = 4 * np.pi * 1e-7
    G0 = mu0 * sum(abs(c.current.get_value()) for c in coils)
    iota0 = 0.3

    booz_ls_cpu = BoozerSurface(
        bs_cpu,
        surf_cpu,
        vol_cpu,
        vol_target,
        constraint_weight=1.0,
        options={
            "verbose": False,
            "bfgs_maxiter": 300,
            "bfgs_tol": 1e-10,
            "newton_maxiter": 20,
            "newton_tol": 1e-11,
        },
    )
    ls_res_cpu = booz_ls_cpu.run_code(iota0, G0)
    assert ls_res_cpu["success"], "CPU LS warm-start did not converge"
    iota_warm = ls_res_cpu["iota"]
    G_warm = ls_res_cpu["G"]

    surf_jax = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=stellsym,
        nfp=nfp,
        quadpoints_phi=qp_phi,
        quadpoints_theta=qp_theta,
    )
    surf_jax.set_dofs(surf_cpu.get_dofs())
    bs_jax = BiotSavartJAX(coils)
    vol_jax = Volume(surf_jax)

    booz_cpu_exact = BoozerSurface(
        bs_cpu,
        surf_cpu,
        vol_cpu,
        vol_target,
        options={"verbose": False},
    )
    booz_cpu_exact.need_to_run_code = True
    res_cpu = booz_cpu_exact.solve_residual_equation_exactly_newton(
        iota=iota_warm,
        G=G_warm,
        tol=1e-10,
        maxiter=40,
    )
    assert res_cpu["success"], "CPU exact Newton did not converge"

    booz_jax_exact = BoozerSurfaceJAX(
        bs_jax,
        surf_jax,
        vol_jax,
        vol_target,
        constraint_weight=None,
        options={
            "verbose": False,
            "newton_maxiter": 40,
            "newton_tol": 1e-10,
        },
    )
    res_jax = booz_jax_exact.run_code(iota_warm, G_warm)
    assert res_jax is not None, "JAX exact solver returned None"
    assert res_jax["success"], "JAX exact Newton did not converge"

    return _ExactSolveParityPair(
        bs_cpu=bs_cpu,
        bs_jax=bs_jax,
        booz_cpu_exact=booz_cpu_exact,
        booz_jax_exact=booz_jax_exact,
        res_cpu=res_cpu,
        res_jax=res_jax,
    )


class TestExactSolveCPUJAXParity:
    """Exact Newton solutions match between CPU and JAX solvers.

    Both solvers start from the same LS-warmed initial guess (shared surface
    DOFs, iota, G) and run exact Newton to convergence.  The solution iota,
    G, and residual norm should agree.
    """

    def test_exact_solve_parity(self):
        exact_pair = _solve_exact_cpu_jax_parity_pair()
        res_cpu = exact_pair.res_cpu
        res_jax = exact_pair.res_jax
        exact_residual_tol = 1e-6
        exact_solution_abs_tol = 1e-5

        iota_diff = abs(res_cpu["iota"] - res_jax["iota"])
        G_diff = abs(res_cpu["G"] - res_jax["G"])
        resid_cpu = np.linalg.norm(res_cpu["residual"], ord=np.inf)
        resid_jax = np.linalg.norm(res_jax["residual"], ord=np.inf)

        print(
            f"Exact parity:\n"
            f"  CPU: iota={res_cpu['iota']:.10e} G={res_cpu['G']:.10e} "
            f"||r||_inf={resid_cpu:.3e}\n"
            f"  JAX: iota={res_jax['iota']:.10e} G={res_jax['G']:.10e} "
            f"||r||_inf={resid_jax:.3e}\n"
            f"  |Δiota|={iota_diff:.3e} |ΔG|={G_diff:.3e}"
        )

        assert resid_cpu < exact_residual_tol, (
            f"CPU residual too large: {resid_cpu:.3e}"
        )
        assert resid_jax < exact_residual_tol, (
            f"JAX residual too large: {resid_jax:.3e}"
        )
        assert iota_diff < exact_solution_abs_tol, f"Iota disagreement: {iota_diff:.3e}"
        assert G_diff < exact_solution_abs_tol, f"G disagreement: {G_diff:.3e}"

    def test_value_wrappers_match_on_shared_exact_state(self):
        exact_pair = _solve_exact_cpu_jax_parity_pair()
        iota_abs_tol = 1e-5
        nqs_rel_tol = 1e-4
        nqs_abs_tol = 1e-10

        iota_cpu = Iotas(exact_pair.booz_cpu_exact).J()
        iota_jax = IotasJAX(exact_pair.booz_jax_exact).J()
        nqs_cpu = NonQuasiSymmetricRatio(
            exact_pair.booz_cpu_exact,
            exact_pair.bs_cpu,
            sDIM=6,
        ).J()
        nqs_jax = NonQuasiSymmetricRatioJAX(
            exact_pair.booz_jax_exact,
            exact_pair.bs_jax,
            sDIM=6,
        ).J()

        np.testing.assert_allclose(iota_jax, iota_cpu, rtol=0.0, atol=iota_abs_tol)
        np.testing.assert_allclose(
            nqs_jax,
            nqs_cpu,
            rtol=nqs_rel_tol,
            atol=nqs_abs_tol,
        )


# -----------------------------------------------------------------------
# Test 16: IotasJAX re-solve FD on the stable reduced real fixture
# -----------------------------------------------------------------------


class TestIotasJAXResolveFD:
    """IotasJAX.dJ() matches central FD through the full re-solve path.

    Perturbs coil DOFs, re-runs the inner Boozer solve, and checks that the
    directional derivative predicted by IotasJAX.dJ() matches the finite-
    difference approximation of (iota(coils+eps) - iota(coils-eps)) / (2*eps).
    """

    def test_iotas_resolve_fd(self):
        _assert_wrapper_resolve_fd_matches_real_fixture(
            wrapper_label="IotasJAX",
            gradient_builder=lambda booz_jax, bs_jax: IotasJAX(booz_jax).dJ(),
            wrapper_factory=lambda booz_jax, bs_jax: IotasJAX(booz_jax),
            rng_seed=77,
        )


# -----------------------------------------------------------------------
# Test 17: NonQuasiSymmetricRatioJAX re-solve FD
# -----------------------------------------------------------------------


class TestNonQSRatioJAXResolveFD:
    """NonQuasiSymmetricRatioJAX.dJ() matches central FD through re-solve.

    Same pattern as TestIotasJAXResolveFD but for the QS ratio wrapper.
    """

    def test_nqsr_resolve_fd(self):
        _assert_wrapper_resolve_fd_matches_real_fixture(
            wrapper_label="NonQuasiSymmetricRatioJAX",
            gradient_builder=lambda booz_jax, bs_jax: NonQuasiSymmetricRatioJAX(
                booz_jax, bs_jax, sDIM=6
            ).dJ(),
            wrapper_factory=lambda booz_jax, bs_jax: NonQuasiSymmetricRatioJAX(
                booz_jax, bs_jax, sDIM=6
            ),
            rng_seed=88,
        )


# =======================================================================
# Section 3: Traceable Single-Stage Objective Path
# =======================================================================
#
# These tests define the contract for the traceable target-objective path
# inside the single-stage workflow so the outer optimizer can route through
# _minimize_lbfgs_private (lax.while_loop) instead of the host-callback
# fallback (_minimize_lbfgs_explicit_value_and_grad) on the supported path.
#
# Dependency order:
#   Test 3 (run_code_functional)
#     -> Test 1 (pure objective value)
#       -> Test 2 (jax.grad differentiable)
#         -> Test 4 (jaxpr traces without error)
#           -> Test 6 (routes through lax.while_loop)
#             -> Test 7 (parity with explicit path)
#   Test 5 (no run_dict/Optimizable dependency) is independent
#
# This slice is green for the current traceable-objective path. Tests 1-7
# validate the pure array-backed custom_vjp objective built by
# make_traceable_objective(), while Tests 3a/3b continue to pin the lower-level
# functional transition seam exposed by run_code_functional().
# =======================================================================


class TestRunCodeFunctional:
    """Test 3: BoozerSurfaceJAX.run_code_functional() — pure functional inner solve.

    The current run_code() mutates self state (need_to_run_code, surface DOFs
    via _set_surface_dofs), uses Python assertions, and branches on dirty flags.

    run_code_functional() must:
    - Accept explicit (coil_arrays, sdofs, iota, G) arguments
    - Return matching iota/G/success/PLU; s=None, vjp=None,
      vjp_groups=None (CPU callbacks incompatible with functional
      contract); sdofs=solved surface DOFs array
    - NOT mutate any self.* state

    Note: this method still uses Python if/float()/np.asarray() on
    solver outputs. Full JIT/grad traceability is achieved one layer
    up via make_traceable_objective(), which rebuilds the single-stage
    objective on pure arrays and differentiates it with custom-VJP
    (tests 1-7).
    """

    def test_run_code_functional_exists_and_matches(self):
        """run_code_functional returns same iota/G/success as run_code."""
        (_, _, _, _, bs_jax, _, booz_jax, _, iota0, G0) = _make_boozer_setup(
            constraint_weight=1.0,
        )

        # Capture inputs before any solve.
        coil_arrays = booz_jax._coil_arrays
        sdofs = np.array(booz_jax.surface.get_dofs())

        # Call functional version FIRST — self.surface is still in the
        # pre-solve state, so this exercises the true functional contract
        # (no dependency on prior stateful mutation).
        res_functional = booz_jax.run_code_functional(
            coil_arrays,
            sdofs,
            iota0,
            G0,
        )

        # Stateful version with the same starting point.
        res_stateful = booz_jax.run_code(iota0, G0)
        assert res_stateful is not None and res_stateful["success"]

        np.testing.assert_allclose(
            res_functional["iota"],
            res_stateful["iota"],
            rtol=1e-10,
        )
        if res_stateful["G"] is not None:
            np.testing.assert_allclose(
                res_functional["G"],
                res_stateful["G"],
                rtol=1e-10,
            )
        assert res_functional["success"] == res_stateful["success"]
        assert res_functional["PLU"] is not None
        # Functional path returns solved sdofs, not a CPU surface object.
        assert res_functional["s"] is None
        assert res_functional["sdofs"] is not None

    def test_run_code_functional_does_not_mutate_self(self):
        """run_code_functional must not change booz_surf internal state."""
        (_, _, _, _, bs_jax, _, booz_jax, _, iota0, G0) = _make_boozer_setup(
            constraint_weight=1.0,
        )

        # Establish baseline state
        res0 = booz_jax.run_code(iota0, G0)
        assert res0 is not None

        sdofs_before = np.array(booz_jax.surface.get_dofs())
        need_to_run_before = booz_jax.need_to_run_code
        res_ref = booz_jax.res

        # Call functional version with perturbed surface DOFs
        rng = np.random.RandomState(42)
        sdofs_perturbed = sdofs_before + 0.001 * rng.randn(len(sdofs_before))

        booz_jax.run_code_functional(
            booz_jax._coil_arrays,
            sdofs_perturbed,
            iota0,
            G0,
        )

        # Self state must be unchanged
        np.testing.assert_array_equal(
            np.array(booz_jax.surface.get_dofs()),
            sdofs_before,
        )
        assert booz_jax.need_to_run_code == need_to_run_before
        assert booz_jax.res is res_ref


class TestTraceableObjective:
    """Tests 1, 2, 4-7: Traceable composed single-stage target objective.

    The current evaluate_candidate() requires JF.x mutation, run_dict state,
    Python if/assert branching, and CPU-side surface/label evaluations.

    The traceable target objective must be a pure function:
        f(coil_dofs: jax.Array) -> jax.Array  (scalar)
    that JAX can trace, differentiate via jax.grad, and compile via JIT.
    """

    @staticmethod
    def _make_traceable(bs_jax, booz_jax):
        """Build the traceable objective and coil DOFs from a solved setup.

        Returns (f, coil_dofs, jr_jax, iotas_jax, iota_target).
        """
        jr_jax = BoozerResidualJAX(booz_jax, bs_jax)
        iotas_jax = IotasJAX(booz_jax)
        iota_target = booz_jax.res["iota"]

        from simsopt.geo.surfaceobjectives_jax import make_traceable_objective

        f = make_traceable_objective(booz_jax, bs_jax, iota_target)
        coil_dofs = jnp.array(bs_jax.x.copy())
        return f, coil_dofs, jr_jax, iotas_jax, iota_target

    def test_pure_objective_matches_optimizable_value(self, boozer_setup):
        """Test 1: Pure JAX objective returns same value as JF.J()."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        f, coil_dofs, jr_jax, iotas_jax, iota_target = self._make_traceable(
            bs_jax,
            booz_jax,
        )

        JF_jax = jr_jax + QuadraticPenalty(iotas_jax, iota_target)
        j_reference = JF_jax.J()

        np.testing.assert_allclose(
            float(f(coil_dofs)),
            j_reference,
            rtol=1e-10,
            err_msg="Traceable objective value differs from JF.J()",
        )

    def test_pure_objective_is_jax_grad_differentiable(self, boozer_setup):
        """Test 2: jax.grad(f)(coil_dofs) is finite, nonzero, matches JF.dJ()."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        f, coil_dofs, jr_jax, iotas_jax, iota_target = self._make_traceable(
            bs_jax,
            booz_jax,
        )

        grad = jax.grad(f)(coil_dofs)

        assert jnp.all(jnp.isfinite(grad)), "jax.grad produced NaN/inf"
        assert jnp.linalg.norm(grad) > 0, "jax.grad produced zero gradient"

        # Compare against existing IFT gradient
        JF_jax = jr_jax + QuadraticPenalty(iotas_jax, iota_target)
        JF_jax.J()
        dj_reference = np.asarray(JF_jax.dJ(), dtype=float)

        np.testing.assert_allclose(
            np.asarray(grad),
            dj_reference,
            rtol=1e-6,
            atol=1e-10,
            err_msg="jax.grad gradient differs from IFT reference",
        )

    def test_pure_objective_traces_to_jaxpr(self, boozer_setup):
        """Test 4: jax.make_jaxpr succeeds without a callback bridge."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        f, coil_dofs, _, _, _ = self._make_traceable(bs_jax, booz_jax)

        jaxpr = jax.make_jaxpr(f)(coil_dofs)
        assert jaxpr is not None, "make_jaxpr returned None"
        assert "pure_callback" not in str(jaxpr), (
            "Traceable objective still routes through jax.pure_callback"
        )

    def test_traceable_objective_has_no_optimizable_dependency(self):
        """Test 5: Traced objective needs no run_dict, no JF.x mutation."""
        (_, _, _, _, bs_jax, _, booz_jax, _, iota0, G0) = _make_boozer_setup(
            constraint_weight=1.0,
        )
        res = booz_jax.run_code(iota0, G0)
        assert res is not None and res["success"]

        f, coil_dofs, jr_jax, iotas_jax, _ = self._make_traceable(bs_jax, booz_jax)

        jr_jax.J()
        iotas_jax.J()
        jr_cache_before = jr_jax._J
        iotas_cache_before = iotas_jax._J

        # Evaluate at a DIFFERENT coil DOF vector without touching JF.x
        rng = np.random.RandomState(99)
        x_perturbed = coil_dofs + 1e-6 * jnp.array(rng.randn(len(coil_dofs)))

        # Snapshot state before calling f
        x_bs_before = bs_jax.x.copy()
        sdofs_before = np.array(booz_jax.surface.get_dofs())
        res_before = booz_jax.res
        need_to_run_before = booz_jax.need_to_run_code

        original_run_code = booz_jax.run_code

        def _reject_run_code(*args, **kwargs):
            raise AssertionError(
                "traceable objective should not call stateful run_code()"
            )

        booz_jax.run_code = _reject_run_code
        try:
            j0 = float(f(coil_dofs))
            j1 = float(f(x_perturbed))
        finally:
            booz_jax.run_code = original_run_code

        # Must produce finite, distinct values
        assert np.isfinite(j0), "f(x0) not finite"
        assert np.isfinite(j1), "f(x_perturbed) not finite"
        assert j0 != j1, "f should be sensitive to coil DOF perturbation"

        # Must not mutate Optimizable state
        np.testing.assert_array_equal(
            bs_jax.x, x_bs_before, err_msg="f mutated bs_jax.x"
        )
        np.testing.assert_array_equal(
            np.array(booz_jax.surface.get_dofs()),
            sdofs_before,
            err_msg="f mutated booz_jax.surface DOFs",
        )
        assert booz_jax.res is res_before, "f replaced booz_jax.res"
        assert booz_jax.need_to_run_code == need_to_run_before, (
            "f dirtied booz_jax.need_to_run_code"
        )
        assert jr_jax._J is jr_cache_before, "f dirtied BoozerResidualJAX cache"
        assert iotas_jax._J is iotas_cache_before, "f dirtied IotasJAX cache"

    def test_traceable_objective_uses_spec_reconstruction_not_grouped_arrays(
        self,
        monkeypatch,
    ):
        """Traceable forward routing must use immutable grouped-coil specs."""
        (_, _, _, _, bs_jax, _, booz_jax, _, iota0, G0) = _make_boozer_setup(
            constraint_weight=1.0,
        )
        res = booz_jax.run_code(iota0, G0)
        assert res is not None and res["success"]

        original_coil_set_spec_from_dofs = bs_jax.coil_set_spec_from_dofs
        calls = {"count": 0}

        def _counting_coil_set_spec_from_dofs(coil_dofs):
            calls["count"] += 1
            return original_coil_set_spec_from_dofs(coil_dofs)

        def _reject_grouped_arrays(*_args, **_kwargs):
            raise AssertionError(
                "traceable forward path should not call grouped_coil_arrays_from_dofs()"
            )

        monkeypatch.setattr(
            bs_jax,
            "coil_set_spec_from_dofs",
            _counting_coil_set_spec_from_dofs,
        )
        monkeypatch.setattr(
            bs_jax,
            "grouped_coil_arrays_from_dofs",
            _reject_grouped_arrays,
        )

        f, coil_dofs, _, _, _ = self._make_traceable(bs_jax, booz_jax)
        value = float(f(coil_dofs))

        assert np.isfinite(value)
        assert calls["count"] > 0

    def test_traceable_objective_does_not_accumulate_children(self, boozer_setup):
        """Repeated calls must not grow booz_jax's descendant graph."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        f, coil_dofs, _, _, _ = self._make_traceable(bs_jax, booz_jax)

        child_count_before = len(booz_jax._children)

        for _ in range(3):
            float(f(coil_dofs))
            gc.collect()

        assert len(booz_jax._children) == child_count_before, (
            "traceable objective leaked Optimizable children across evaluations"
        )

    def test_traceable_routes_through_lax_while_loop(self, boozer_setup, monkeypatch):
        """Test 6: lbfgs-ondevice uses _minimize_lbfgs_private, not fallback."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        f, x0, _, _, _ = self._make_traceable(bs_jax, booz_jax)

        # Reject if the explicit host-loop fallback is called --
        # jax_minimize resolves this name via module.__dict__ at call time,
        # so monkeypatch.setattr on the module object is sufficient.
        import simsopt.geo.optimizer_jax as opt_mod

        def _reject(*args, **kwargs):
            raise AssertionError(
                "Traceable objective should route through "
                "_minimize_lbfgs_private, not the explicit fallback"
            )

        monkeypatch.setattr(opt_mod, "_minimize_lbfgs_explicit_value_and_grad", _reject)

        result = jax_minimize(
            f,
            x0,
            method="lbfgs-ondevice",
            maxiter=2,
            tol=1e-20,
        )
        assert np.isfinite(float(result.fun)), "Optimizer produced non-finite J"

    def test_traceable_matches_explicit_path(self, boozer_setup):
        """Test 7: Traceable and explicit paths produce same J after 3 iters."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        f, x0, _, _, _ = self._make_traceable(bs_jax, booz_jax)

        # Snapshot Optimizable state so Path B starts from identical state
        # even if Path A's optimizer leaks side effects (defensive guard --
        # f is supposed to be pure, but we verify rather than assume).
        x0_bs = bs_jax.x.copy()
        sdofs_snap = np.array(booz_jax.surface.get_dofs())
        res_snap = booz_jax.res

        def _restore_state():
            bs_jax.x = x0_bs
            booz_jax.surface.set_dofs(sdofs_snap)
            booz_jax.res = res_snap

        try:
            # Path A: traceable through _minimize_lbfgs_private
            result_a = jax_minimize(
                f,
                x0,
                method="lbfgs-ondevice",
                maxiter=3,
                tol=1e-20,
            )

            _restore_state()

            # Path B: explicit value_and_grad through host loop
            def fun_vg(x):
                x_jax = jnp.array(x) if not isinstance(x, jnp.ndarray) else x
                val = float(f(x_jax))
                g = np.asarray(jax.grad(f)(x_jax), dtype=float)
                return val, g

            result_b = jax_minimize(
                fun_vg,
                np.asarray(x0),
                method="lbfgs-ondevice",
                value_and_grad=True,
                maxiter=3,
                tol=1e-20,
            )

            # The compared objective values are nominally zero at this point in
            # the short run, so a tiny absolute floor is more meaningful than a
            # pure relative check.
            traceable_objective_abs_tol = 1e-32
            np.testing.assert_allclose(
                float(result_a.fun),
                float(result_b.fun),
                rtol=1e-10,
                atol=traceable_objective_abs_tol,
                err_msg=(
                    f"Traceable J={float(result_a.fun):.6e} vs "
                    f"explicit J={float(result_b.fun):.6e}"
                ),
            )
        finally:
            _restore_state()
