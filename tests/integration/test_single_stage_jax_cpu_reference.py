"""
Single-stage JAX backend integration tests (Milestone 5).

Validates:
1. BoozerResidualJAX.J() is small at converged surface (both CPU and JAX).
2. IotasJAX.J() is finite at independently converged solutions.
3. NonQuasiSymmetricRatioJAX.J() is finite and non-negative.
4. Adjoint-solve consistency (H^T adj = dJ_ds).
5. VJP produces finite, non-zero derivative.
6. Reduced real-fixture on-device wrappers match CPU reference values and gradients.
7. Fixed-surface and re-solve FD validate the composed JAX gradient path.
8. Composite objective value and gradient are finite and non-zero.
9. Backend selection constructs correct object types.

Gradient validation uses two complementary lanes:
- direct CPU-vs-JAX parity on the shared reduced real-fixture CPU-reference vs JAX-ondevice path
- finite-difference checks on the JAX fixed-surface and re-solve paths

The FD checks remain necessary for the full re-solve lane, where CPU and JAX
can use different internal linear algebra while still targeting the same
outer objective.

Pure JAX helper-path coverage lives in ``test_single_stage_jax.py``.
This file keeps the heavier CPU-reference integration lanes together and
therefore requires ``simsoptpp``.
"""

import gc
import logging
import re
from functools import partial
import types

logger = logging.getLogger(__name__)

import pytest
from conftest import (
    assert_arrays_on_device,
    enable_non_strict_jax_backend,
    enable_strict_jax_backend,
    parity_device,
    relative_error,
)
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
    CoilSetDofExtractionSpec,
    CoilSpec,
    CurveCWSFourierRZSpec,
    CurveFilamentSpec,
    CurveHelicalSpec,
    CurvePlanarFourierSpec,
    CurvePerturbedSpec,
    FieldEvalSpec,
    GroupedCoilSetSpec,
    curve_gamma_and_dash_from_dofs,
    curve_gamma_and_dash_from_spec,
    curve_geometry_from_dofs,
    curve_pullback_from_spec,
    curve_spec_from_curve,
    grouped_coil_set_spec_from_coil_specs,
    make_optimizable_dof_map_spec,
)
from simsopt.jax_core.curve_geometry import (  # noqa: E402
    _mapped_full_dofs,
    _mapped_input_dofs,
)
from simsopt.geo.boozersurface_jax import (  # noqa: E402
    BoozerSurfaceJAX,
    _boozer_ls_coil_vjp,
    _boozer_ls_coil_vjp_groups,
    _ls_decision_vector,
    _make_ls_penalty_objective,
    _select_exact_residual_fn,
)
from simsopt.geo._boozersurface_current_guard import (  # noqa: E402
    _none_G_coil_gradient_error,
)
from simsopt.geo.optimizer_jax import (  # noqa: E402
    PRIVATE_OPTIMIZER_JAX_VERSION,
    jax_minimize,
    private_optimizer_runtime_is_supported,
)
from simsopt.geo.surfaceobjectives_jax import (  # noqa: E402
    BoozerResidualJAX,
    IotasJAX,
    NonQuasiSymmetricRatioJAX,
    _boozer_residual_J_of_x_inner,
    _qs_ratio_pure,
    compute_standard_surface_objective_gradients,
)
from simsopt.geo.curve import Curve, RotatedCurve  # noqa: E402

from examples.single_stage_optimization.SINGLE_STAGE import (  # noqa: E402
    single_stage_banana_example as single_stage_example,
)

_HIDDEN_SPEC_FALLBACK_PATTERN = (
    "BiotSavartJAX.*hidden immutable-spec compatibility fallback.*"
)
_HIDDEN_SPEC_WARNING_PATTERN = (
    "BiotSavartJAX.*hidden immutable-spec compatibility fallback.*legacy adapter seam"
)
_REMOVED_LIVE_GRAPH_SPEC_SEAM_PATTERN = (
    "BiotSavartJAX.*coil_set_spec\\(\\).*legacy adapter seam via "
    "live-graph geometry extraction was removed"
)
_PUBLIC_COIL_VJP_WARNING_PATTERN = (
    "BiotSavartJAX.*public CPU coil\\.vjp\\(\\) pullback compatibility "
    "path.*legacy adapter seam"
)
_PUBLIC_COIL_VJP_STRICT_PATTERN = (
    "BiotSavartJAX.*public CPU coil\\.vjp\\(\\) pullback compatibility "
    "path.*strict=True"
)
# Solved-state objective parity can land in the ~1e-24 range, where tiny
# evaluation-order differences need a small absolute floor to stay meaningful.
_TRACEABLE_OBJECTIVE_ABS_TOL = 1e-28
_REAL_FIXTURE_SOLVER_CPU_JAX_TOLS = {
    "objective": (1e-12, 1e-18),
    "residual_inf": (1e-12, 1e-12),
    "iota": (0.0, 1e-12),
    "G": (0.0, 1e-12),
    "label_value": (1e-12, 1e-12),
    "label_error": (1e-12, 1e-12),
    "axis_z_abs": (0.0, 1e-12),
}
_REAL_FIXTURE_SOLVER_CPU_GPU_TOLS = {
    "objective": (1e-8, 1e-16),
    "residual_inf": (1e-8, 1e-10),
    "iota": (0.0, 1e-10),
    "G": (0.0, 1e-10),
    "label_value": (1e-10, 1e-10),
    "label_error": (1e-10, 1e-10),
    "axis_z_abs": (0.0, 1e-10),
}
_EXACT_SOLVER_END_STATE_TOLS = {
    "objective": (1e-10, 1e-24),
    "residual_inf": (1e-10, 1e-12),
    "iota": (0.0, 1e-12),
    "G": (0.0, 1e-12),
    "label_value": (1e-12, 1e-12),
    "label_error": (1e-12, 1e-12),
    "axis_z_abs": (0.0, 1e-12),
}
_EXACT_SOLVER_RESIDUAL_INF_MAX = 1e-12


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def test_single_stage_curvature_threshold_policy_caps_at_40():
    assert single_stage_example.resolve_curvature_threshold(10.0) == pytest.approx(20.0)
    assert single_stage_example.resolve_curvature_threshold(30.0) == pytest.approx(30.0)
    assert single_stage_example.resolve_curvature_threshold(80.0) == pytest.approx(40.0)


def _iota_unit_rhs(plu):
    """Return the standard IotasJAX inner cotangent for the LS path."""
    n = plu[1].shape[0]
    rhs = np.zeros(n)
    rhs[-2] = 1.0
    return rhs


def _build_boozer_wrapper(wrapper_name, booz_surf, biotsavart):
    """Instantiate a CPU or JAX Boozer wrapper by test label."""
    builders = {
        "BoozerResidual": lambda: BoozerResidual(booz_surf, biotsavart),
        "Iotas": lambda: Iotas(booz_surf),
        "NonQuasiSymmetricRatio": lambda: NonQuasiSymmetricRatio(
            booz_surf, biotsavart, sDIM=6
        ),
        "BoozerResidualJAX": lambda: BoozerResidualJAX(booz_surf, biotsavart),
        "IotasJAX": lambda: IotasJAX(booz_surf),
        "NonQuasiSymmetricRatioJAX": lambda: NonQuasiSymmetricRatioJAX(
            booz_surf, biotsavart, sDIM=6
        ),
    }
    return builders[wrapper_name]()


def _make_guarded_gradient_failure(component):
    """Return a callback that raises the canonical none-G gradient error."""

    def reject_gradient(*args, **kwargs):
        del args, kwargs
        raise ValueError(_none_G_coil_gradient_error(component))

    return reject_gradient


_enable_strict_jax_backend = partial(enable_strict_jax_backend, mode="jax_gpu_parity")
_enable_non_strict_jax_backend = partial(
    enable_non_strict_jax_backend,
    mode="jax_gpu_parity",
)
_enable_fast_non_strict_jax_backend = partial(
    enable_non_strict_jax_backend,
    mode="jax_gpu_fast",
)


def _assert_hidden_spec_fallback_rejected(
    monkeypatch,
    request,
    callback,
    *,
    api_name,
    mode="jax_gpu_parity",
):
    _enable_strict_jax_backend(monkeypatch, request, mode=mode)
    with pytest.raises(
        RuntimeError,
        match=_HIDDEN_SPEC_FALLBACK_PATTERN
        + rf".*{re.escape(api_name)}\(\).*strict=True",
    ):
        callback()


def _assert_hidden_spec_fallback_warns(
    monkeypatch,
    request,
    callback,
    *,
    api_name,
    mode="jax_gpu_parity",
):
    _enable_non_strict_jax_backend(monkeypatch, request, mode=mode)
    with pytest.warns(
        RuntimeWarning,
        match=(
            _HIDDEN_SPEC_WARNING_PATTERN.replace(".*legacy adapter seam", "")
            + rf".*{re.escape(api_name)}\(\).*legacy adapter seam"
        ),
    ):
        return callback()


def _assert_removed_live_graph_spec_seam(
    monkeypatch,
    request,
    callback,
    *,
    mode,
):
    if mode == "strict":
        _enable_strict_jax_backend(monkeypatch, request)
    else:
        _enable_non_strict_jax_backend(monkeypatch, request)
    with pytest.raises(
        RuntimeError,
        match=_REMOVED_LIVE_GRAPH_SPEC_SEAM_PATTERN,
    ):
        callback()


def _make_biotsavart_jax_for_coils(coils):
    bs_jax = object.__new__(BiotSavartJAX)
    bs_jax._coils = coils
    return bs_jax


def _single_coil_cotangent_arrays(d_gamma, d_gammadash, d_current):
    return [
        (
            jnp.asarray([d_gamma]),
            jnp.asarray([d_gammadash]),
            jnp.asarray([d_current]),
        )
    ]


def _assert_grouped_coil_set_spec_allclose(observed, expected, *, atol=1e-12):
    assert isinstance(observed, GroupedCoilSetSpec)
    assert len(observed.groups) == len(expected.groups)
    for observed_group, expected_group in zip(observed.groups, expected.groups):
        assert observed_group.coil_indices == expected_group.coil_indices
        np.testing.assert_allclose(
            np.asarray(observed_group.gammas),
            np.asarray(expected_group.gammas),
            atol=atol,
        )
        np.testing.assert_allclose(
            np.asarray(observed_group.gammadashs),
            np.asarray(expected_group.gammadashs),
            atol=atol,
        )
        np.testing.assert_allclose(
            np.asarray(observed_group.currents),
            np.asarray(expected_group.currents),
            atol=atol,
        )


def _assert_grouped_field_data_matches_spec(observed, expected, *, atol=1e-12):
    assert len(observed) == len(expected.groups)
    for observed_group, expected_group in zip(observed, expected.groups):
        observed_gamma, observed_gammadash, observed_current, observed_indices = (
            observed_group
        )
        np.testing.assert_allclose(
            observed_gamma,
            np.asarray(expected_group.gammas),
            atol=atol,
        )
        np.testing.assert_allclose(
            observed_gammadash,
            np.asarray(expected_group.gammadashs),
            atol=atol,
        )
        np.testing.assert_allclose(
            observed_current,
            np.asarray(expected_group.currents),
            atol=atol,
        )
        assert observed_indices == list(expected_group.coil_indices)


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


def _make_fixed_state_exact_directional_objective(
    booz_surf,
    biotsavart,
    lm,
    iota,
    G,
):
    """Build ``coil_dofs -> <lm, r_exact(x*, coil_dofs)>`` at a frozen exact state."""
    exact_state = jnp.concatenate([booz_surf._get_surface_dofs(), jnp.array([iota, G])])
    residual_fn = _select_exact_residual_fn(booz_surf.stellsym)
    lm_jax = jnp.asarray(lm)
    residual_kwargs = {
        "quadpoints_phi": booz_surf.quadpoints_phi,
        "quadpoints_theta": booz_surf.quadpoints_theta,
        "mpol": booz_surf.mpol,
        "ntor": booz_surf.ntor,
        "nfp": booz_surf.nfp,
        "stellsym": booz_surf.stellsym,
        "scatter_indices": booz_surf.scatter_indices,
        "surface_kind": booz_surf._surface_geometry_kind,
        "targetlabel": booz_surf.targetlabel,
        "label_type": booz_surf.label_type,
        "phi_idx": booz_surf.phi_idx,
        "mask_indices": booz_surf._compute_stellsym_mask_indices(),
        "weight_inv_modB": booz_surf.options["weight_inv_modB"],
    }

    def directional_objective(coil_dofs):
        return jnp.vdot(
            lm_jax,
            residual_fn(
                exact_state,
                coil_set_spec=biotsavart.coil_set_spec_from_dofs(coil_dofs),
                **residual_kwargs,
            ),
        )

    return directional_objective


def _assert_directional_derivative_matches_fd(
    full_gradient,
    directional_objective_at,
    x0,
    *,
    rng_seed,
    eps,
    num_directions,
    rel_tol,
    abs_tol,
    label,
):
    rng = np.random.RandomState(rng_seed)

    for i in range(num_directions):
        direction = rng.randn(len(x0))
        direction /= np.linalg.norm(direction)

        dd_vjp = float(np.dot(full_gradient, direction))
        dd_fd = float(
            (
                directional_objective_at(x0 + eps * direction)
                - directional_objective_at(x0 - eps * direction)
            )
            / (2.0 * eps)
        )

        abs_err = abs(dd_vjp - dd_fd)
        rel_err = abs_err / (abs(dd_fd) + 1e-30)
        assert rel_err < rel_tol or abs_err < abs_tol, (
            f"{label}[{i}]: vjp={dd_vjp:.6e} fd={dd_fd:.6e} "
            f"rel={rel_err:.2e} abs={abs_err:.2e}"
        )


def _assert_gradients_finite_nonzero(gradients, message_prefix):
    for grad in gradients:
        assert np.all(np.isfinite(grad)), f"{message_prefix} produced NaN/inf"
        assert np.linalg.norm(grad) > 0, f"{message_prefix} produced zero gradient"


def _cpu_single_stage_wrapper_gradients(booz_cpu, bs_cpu):
    return [
        np.array(BoozerResidual(booz_cpu, bs_cpu).dJ()),
        np.array(Iotas(booz_cpu).dJ()),
        np.array(NonQuasiSymmetricRatio(booz_cpu, bs_cpu, sDIM=6).dJ()),
    ]


def _cpu_single_stage_wrapper_values(booz_cpu, bs_cpu):
    return [
        BoozerResidual(booz_cpu, bs_cpu).J(),
        Iotas(booz_cpu).J(),
        NonQuasiSymmetricRatio(booz_cpu, bs_cpu, sDIM=6).J(),
    ]


def _jax_single_stage_wrapper_values(booz_jax, bs_jax):
    return [
        BoozerResidualJAX(booz_jax, bs_jax).J(),
        IotasJAX(booz_jax).J(),
        NonQuasiSymmetricRatioJAX(booz_jax, bs_jax, sDIM=6).J(),
    ]


def _make_jax_standard_wrapper_triplet(booz_jax, bs_jax):
    return (
        BoozerResidualJAX(booz_jax, bs_jax),
        IotasJAX(booz_jax),
        NonQuasiSymmetricRatioJAX(booz_jax, bs_jax, sDIM=6),
    )


def _jax_single_stage_wrapper_gradients(booz_jax, bs_jax):
    boozer_residual, iotas, non_qs_ratio = _make_jax_standard_wrapper_triplet(
        booz_jax,
        bs_jax,
    )
    batched_gradients = compute_standard_surface_objective_gradients(
        boozer_residual,
        iotas,
        non_qs_ratio,
    )
    return [np.array(gradient) for gradient in batched_gradients]


def _build_real_fixture_ondevice_m5_pair():
    cpu_fixture = build_real_single_stage_init_fixture(
        backend="cpu",
        optimizer_backend="scipy",
    )
    jax_fixture = build_real_single_stage_init_fixture(
        backend="jax",
        optimizer_backend="ondevice",
    )
    return cpu_fixture, jax_fixture


def _host_metric_array(value):
    if value is None:
        return None
    if isinstance(value, jax.Array):
        value = jax.device_get(jax.block_until_ready(value))
    return np.asarray(value, dtype=float)


def _host_metric_scalar(value):
    return float(np.asarray(_host_metric_array(value), dtype=float))


def _derived_solver_objective(result, residual):
    solve_type = str(result["type"])
    if residual is None:
        return _host_metric_scalar(result["fun"])
    if solve_type == "ls":
        return float(0.5 * np.sum(np.square(residual)))
    if solve_type == "exact":
        return float(0.5 * np.mean(np.square(residual)))
    return _host_metric_scalar(result["fun"])


def _collect_boozer_solver_end_state_metrics(booz):
    result = booz.res
    assert result is not None, "Boozer solver result missing"
    residual = _host_metric_array(result.get("residual"))
    residual_inf = (
        np.inf if residual is None else float(np.linalg.norm(residual, ord=np.inf))
    )
    label_value = float(booz.label.J())
    surface_gamma = np.asarray(booz.surface.gamma(), dtype=float)
    G_value = result.get("G")
    return {
        "success": bool(result.get("success", False)),
        "objective": _derived_solver_objective(result, residual),
        "residual_inf": residual_inf,
        "iota": float(result["iota"]),
        "G": None if G_value is None else _host_metric_scalar(G_value),
        "label_value": label_value,
        "label_error": abs(label_value - float(booz.targetlabel)),
        "axis_z_abs": abs(float(surface_gamma[0, 0, 2])),
        "iterations": int(result["iter"]),
        "solve_type": str(result["type"]),
    }


def _format_boozer_solver_end_state_diagnostics(
    reference_label,
    reference_metrics,
    candidate_label,
    candidate_metrics,
    tolerances,
):
    metric_lines = []
    for metric_name in tolerances:
        reference_value = reference_metrics[metric_name]
        candidate_value = candidate_metrics[metric_name]
        if reference_value is None or candidate_value is None:
            diff_text = "n/a"
        else:
            diff_text = f"|Δ|={abs(candidate_value - reference_value):.3e}"
        metric_lines.append(
            f"{metric_name}: {reference_label}={reference_value!r} "
            f"{candidate_label}={candidate_value!r} {diff_text}"
        )
    envelope_lines = [
        f"{metric}: rtol={rtol:.1e}, atol={atol:.1e}"
        for metric, (rtol, atol) in tolerances.items()
    ]
    return (
        f"{reference_label}: success={reference_metrics['success']} "
        f"iter={reference_metrics['iterations']} type={reference_metrics['solve_type']}\n"
        f"{candidate_label}: success={candidate_metrics['success']} "
        f"iter={candidate_metrics['iterations']} type={candidate_metrics['solve_type']}\n"
        + "\n".join(metric_lines)
        + "\nAccepted drift envelopes: "
        + "; ".join(envelope_lines)
    )


def _assert_boozer_solver_end_state_parity(
    reference_label,
    reference_metrics,
    candidate_label,
    candidate_metrics,
    *,
    tolerances,
    max_reference_residual_inf=None,
    max_candidate_residual_inf=None,
):
    diagnostics = _format_boozer_solver_end_state_diagnostics(
        reference_label,
        reference_metrics,
        candidate_label,
        candidate_metrics,
        tolerances,
    )
    logger.info("Boozer solver end-state diagnostics:\n%s", diagnostics)

    mismatch_reasons = []
    if not reference_metrics["success"]:
        mismatch_reasons.append(f"{reference_label} solve did not converge")
    if not candidate_metrics["success"]:
        mismatch_reasons.append(f"{candidate_label} solve did not converge")

    for metric_name, (rtol, atol) in tolerances.items():
        reference_value = reference_metrics[metric_name]
        candidate_value = candidate_metrics[metric_name]
        if reference_value is None or candidate_value is None:
            if reference_value != candidate_value:
                mismatch_reasons.append(
                    f"{metric_name}: {reference_label}={reference_value!r} "
                    f"{candidate_label}={candidate_value!r}"
                )
            continue
        if not np.isclose(candidate_value, reference_value, rtol=rtol, atol=atol):
            mismatch_reasons.append(
                f"{metric_name}: {reference_label}={reference_value:.15e} "
                f"{candidate_label}={candidate_value:.15e} "
                f"(rtol={rtol:.1e}, atol={atol:.1e})"
            )

    if (
        max_reference_residual_inf is not None
        and reference_metrics["residual_inf"] > max_reference_residual_inf
    ):
        mismatch_reasons.append(
            f"{reference_label} residual_inf={reference_metrics['residual_inf']:.3e} "
            f"exceeds {max_reference_residual_inf:.1e}"
        )
    if (
        max_candidate_residual_inf is not None
        and candidate_metrics["residual_inf"] > max_candidate_residual_inf
    ):
        mismatch_reasons.append(
            f"{candidate_label} residual_inf={candidate_metrics['residual_inf']:.3e} "
            f"exceeds {max_candidate_residual_inf:.1e}"
        )

    if mismatch_reasons:
        pytest.fail(
            "Boozer solver end-state parity failed: "
            + "; ".join(mismatch_reasons)
            + "\n"
            + diagnostics
        )


def _assert_boozer_surfaces_end_state_parity(
    reference_label,
    reference_boozer,
    candidate_label,
    candidate_boozer,
    *,
    tolerances,
    max_reference_residual_inf=None,
    max_candidate_residual_inf=None,
):
    _assert_boozer_solver_end_state_parity(
        reference_label,
        _collect_boozer_solver_end_state_metrics(reference_boozer),
        candidate_label,
        _collect_boozer_solver_end_state_metrics(candidate_boozer),
        tolerances=tolerances,
        max_reference_residual_inf=max_reference_residual_inf,
        max_candidate_residual_inf=max_candidate_residual_inf,
    )


def _build_real_fixture_gpu_solver_pair():
    cpu_fixture = build_real_single_stage_init_fixture(
        backend="cpu",
        optimizer_backend="scipy",
    )
    booz_cpu = cpu_fixture["boozer_surface"]
    cpu_result = booz_cpu.res
    assert cpu_result is not None and cpu_result.get("success", False)

    gpu_fixture = build_real_single_stage_init_fixture(
        backend="jax",
        optimizer_backend="ondevice",
        boozer_surface_dofs_override=np.asarray(
            booz_cpu.surface.get_dofs(),
            dtype=float,
        ),
        boozer_iota_override=float(cpu_result["iota"]),
        boozer_G_override=float(cpu_result["G"]),
    )
    return booz_cpu, gpu_fixture, gpu_fixture["boozer_surface"].res


def _assert_gpu_boozer_solver_result_on_device(gpu, gpu_fixture, gpu_result):
    assert gpu_fixture["boozer_optimizer_backend"] == "ondevice"
    assert gpu_result is not None and gpu_result.get("success", False)
    assert gpu_result["type"] == "ls"
    assert_arrays_on_device(
        gpu,
        gpu_result["jacobian"],
        gpu_result["hessian"],
        *gpu_result["PLU"],
    )


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


class _UnsupportedCurveForRotation(Curve):
    """Minimal curve stub with no public pullback hooks."""

    def __init__(self):
        self.quadpoints = np.array([0.0, 0.5])
        super().__init__(x0=np.array([]))

    def invalidate_cache(self):
        pass


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


class _CpuFallbackRecordingCoil:
    """Coil stub that records use of the public CPU ``coil.vjp()`` contract."""

    def __init__(self, *, rotated=False, phi=np.pi / 2.0):
        curve = _CpuProjectableCurve()
        self.curve = RotatedCurve(curve, phi=phi, flip=False) if rotated else curve
        self.current = _RecordingCurrent()
        self.calls = []

    def vjp(self, dg, dgd, dc):
        dg_arr = np.asarray(dg, dtype=float)
        dgd_arr = np.asarray(dgd, dtype=float)
        dc_arr = np.atleast_1d(np.asarray(dc, dtype=float))
        self.calls.append((dg_arr, dgd_arr, dc_arr))
        curve = self.curve
        if isinstance(curve, RotatedCurve):
            dg_arr = dg_arr @ np.asarray(curve.rotmatT, dtype=float)
            dgd_arr = dgd_arr @ np.asarray(curve.rotmatT, dtype=float)
            curve = curve.curve
        return (
            curve.dgamma_by_dcoeff_vjp(dg_arr)
            + curve.dgammadash_by_dcoeff_vjp(dgd_arr)
            + self.current.vjp(dc_arr)
        )


class _RotatedUnsupportedRecordingCoil:
    """Rotated unsupported coil stub that must be rejected before ``coil.vjp()``."""

    def __init__(self):
        self.curve = RotatedCurve(_UnsupportedCurveForRotation(), np.pi / 4.0, False)
        self.current = _RecordingCurrent()
        self.calls = []

    def vjp(self, dg, dgd, dc):
        self.calls.append((dg, dgd, dc))
        raise AssertionError(
            "Rotated unsupported curves should be rejected before coil.vjp()"
        )


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


def _build_xyz_curve(nquadpoints):
    curve = CurveXYZFourier(nquadpoints, 2)
    curve.set_dofs(
        np.array(
            [
                1.0,
                0.2,
                -0.1,
                0.1,
                -0.05,
                0.03,
                0.8,
                -0.2,
                0.15,
                0.02,
                -0.01,
                0.04,
                -0.03,
                0.02,
                -0.05,
            ]
        )
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
            np.column_stack(
                (
                    -3.2e-3 * np.sin(2.0 * np.pi * quadpoints),
                    2.1e-3 * np.cos(2.0 * np.pi * quadpoints),
                    -1.7e-3 * np.sin(4.0 * np.pi * quadpoints),
                )
            ),
            np.column_stack(
                (
                    -2.8e-3 * np.cos(2.0 * np.pi * quadpoints),
                    -1.3e-3 * np.sin(2.0 * np.pi * quadpoints),
                    3.6e-3 * np.cos(4.0 * np.pi * quadpoints),
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


def _build_surface_bound_curve(curve_cls, nquadpoints):
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

    curve = curve_cls(
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


def _build_surface_bound_cpp_curve(nquadpoints):
    return _build_surface_bound_curve(CurveCWSFourierCPP, nquadpoints)


def _build_surface_bound_jax_curve(nquadpoints):
    return _build_surface_bound_curve(CurveCWSFourier, nquadpoints)


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


def _assert_curve_exposes_immutable_spec(curve, spec_type):
    curve_spec = curve.to_spec()

    assert isinstance(curve_spec, spec_type)
    np.testing.assert_allclose(
        np.asarray(curve_gamma_and_dash_from_spec(curve_spec)[0]),
        curve.gamma(),
        atol=1e-12,
    )


def _assert_coil_set_spec_prefers_immutable_curve_specs(
    monkeypatch,
    curve,
    current_value,
    message,
):
    coil = Coil(curve, Current(current_value))
    bs_jax = BiotSavartJAX([coil])

    monkeypatch.setattr(
        bs_jax,
        "_coil_set_spec_from_dofs_via_grouped_arrays",
        lambda _coil_dofs: (_ for _ in ()).throw(AssertionError(message)),
    )

    coil_set_spec = bs_jax.coil_set_spec_from_dofs(jnp.asarray(bs_jax.x))
    assert isinstance(coil_set_spec, GroupedCoilSetSpec)


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


def _assert_curve_spec_pullback_matches_curve_methods(
    curve,
    *,
    expect_surface: bool = False,
):
    spec = curve_spec_from_curve(curve)
    gamma = jnp.asarray(curve.gamma(), dtype=jnp.float64)
    gammadash = jnp.asarray(curve.gammadash(), dtype=jnp.float64)
    dg = jnp.reshape(
        jnp.linspace(0.1, 1.0, gamma.size, dtype=jnp.float64),
        gamma.shape,
    )
    dgd = jnp.reshape(
        jnp.linspace(-0.7, 0.2, gammadash.size, dtype=jnp.float64),
        gammadash.shape,
    )

    coeff_cotangent, surface_cotangent = curve_pullback_from_spec(spec, dg, dgd)
    curve_dofs = jnp.asarray(
        curve.full_x
        if getattr(curve, "_jax_curve_dof_mode", "local") == "full"
        else curve.get_dofs(),
        dtype=jnp.float64,
    )
    if hasattr(curve, "dgamma_by_dcoeff_vjp_jax") and hasattr(
        curve,
        "dgammadash_by_dcoeff_vjp_jax",
    ):
        coeff_expected = curve.dgamma_by_dcoeff_vjp_jax(
            curve_dofs,
            dg,
        ) + curve.dgammadash_by_dcoeff_vjp_jax(curve_dofs, dgd)
    else:
        from simsopt.geo.curvexyzfourier import jaxfouriercurve_pure

        quadpoints = jnp.asarray(curve.quadpoints, dtype=jnp.float64)
        tangents = jnp.ones_like(quadpoints)

        def outputs(dofs):
            def gamma_kernel(qp):
                return jaxfouriercurve_pure(dofs, qp, curve.order)

            return jax.jvp(gamma_kernel, (quadpoints,), (tangents,))

        _, pullback = jax.vjp(outputs, curve_dofs)
        (coeff_expected,) = pullback((dg, dgd))

    np.testing.assert_allclose(
        np.asarray(coeff_cotangent),
        np.asarray(coeff_expected),
        rtol=1e-10,
        atol=1e-12,
    )

    if not expect_surface:
        assert surface_cotangent is None
        return

    surface_dofs = jnp.asarray(curve.surf.get_dofs(), dtype=jnp.float64)
    surface_expected = curve.dgamma_by_dsurf_vjp_jax(
        surface_dofs,
        dg,
    ) + curve.dgammadash_by_dsurf_vjp_jax(surface_dofs, dgd)
    assert surface_cotangent is not None
    np.testing.assert_allclose(
        np.asarray(surface_cotangent),
        np.asarray(surface_expected),
        rtol=1e-10,
        atol=1e-12,
    )


def _central_difference_gradient(fun, x0, eps):
    grad = np.zeros(x0.shape[0], dtype=float)
    for index in range(x0.shape[0]):
        x_plus = x0.at[index].add(eps)
        x_minus = x0.at[index].add(-eps)
        grad[index] = (float(fun(x_plus)) - float(fun(x_minus))) / (2.0 * eps)
    return grad


def _assert_curve_spec_geometry_matches_live_curve(curve):
    spec = curve.to_spec()
    gamma, gammadash, gammadashdash = curve_geometry_from_dofs(spec, spec.dofs)

    np.testing.assert_allclose(np.asarray(gamma), curve.gamma(), atol=1e-12)
    np.testing.assert_allclose(np.asarray(gammadash), curve.gammadash(), atol=1e-12)
    try:
        reference_gammadashdash = curve.gammadashdash()
    except RuntimeError:
        assert gammadashdash.shape == gamma.shape
        assert np.all(np.isfinite(np.asarray(gammadashdash)))
        return
    np.testing.assert_allclose(
        np.asarray(gammadashdash),
        reference_gammadashdash,
        atol=1e-12,
    )


def _assert_curve_spec_gamma_and_dash_gradient_matches_fd(curve, *, eps=1.0e-7):
    spec = curve.to_spec()
    dofs0 = jnp.asarray(spec.dofs, dtype=jnp.float64)
    gamma0, gammadash0 = curve_gamma_and_dash_from_dofs(spec, dofs0)
    gamma_weights = jnp.reshape(
        jnp.linspace(0.1, 1.1, gamma0.size, dtype=jnp.float64),
        gamma0.shape,
    )
    gammadash_weights = jnp.reshape(
        jnp.linspace(-0.9, 0.3, gammadash0.size, dtype=jnp.float64),
        gammadash0.shape,
    )

    def objective(dofs):
        gamma, gammadash = curve_gamma_and_dash_from_dofs(spec, dofs)
        return jnp.sum(gamma * gamma_weights) + jnp.sum(gammadash * gammadash_weights)

    grad = np.asarray(jax.grad(objective)(dofs0))
    grad_fd = _central_difference_gradient(objective, dofs0, eps)
    np.testing.assert_allclose(grad, grad_fd, rtol=1e-6, atol=1e-9)


_REAL_RESOLVE_FD_ABS_TOL = 1e-8
_REAL_RESOLVE_FD_TAYLOR_RATE = 0.55
_REAL_RESOLVE_FD_EPSILONS = (4.0e-4, 2.0e-4, 1.0e-4)
_REAL_RESOLVE_FD_MIN_STABLE_SAMPLES = 3
_REAL_RESOLVE_FD_MIN_STABLE_EPS = 2
_REAL_RESOLVE_FD_AXIS_DIRECTION_FRACTIONS = (0.0, 0.5, 1.0)
_REAL_RESOLVE_FD_NQSR_SDIM = 6
_STABLE_IOTA_ABS_TOL = 5e-4
_STABLE_G_REL_TOL = 1e-4
_STABLE_FUN_REL_TOL = 1e-2


class _RealResolveFDWrapperSpec(NamedTuple):
    label: str
    gradient_builder: object
    value_builder: object


class _RealResolveFDBaselineState(NamedTuple):
    coil_dofs: np.ndarray
    surface_dofs: np.ndarray
    iota: float
    G: float
    fun: float
    result: dict[str, object]


class _RealResolveFDProbeOutcome(NamedTuple):
    stable: bool
    reason: str
    coil_dofs: np.ndarray | None
    surface_dofs: np.ndarray | None
    iota: float | None
    G: float | None
    weight_inv_modB: bool | None


class _RealResolveFDEpsilonSample(NamedTuple):
    eps: float
    plus: _RealResolveFDProbeOutcome
    minus: _RealResolveFDProbeOutcome


class _RealResolveFDDirectionSample(NamedTuple):
    direction: np.ndarray
    epsilon_samples: tuple[_RealResolveFDEpsilonSample, ...]


class _RealResolveFDSuite(NamedTuple):
    bs_jax: BiotSavartJAX
    booz_jax: BoozerSurfaceJAX
    baseline_state: _RealResolveFDBaselineState
    gradients: dict[str, np.ndarray]
    direction_samples: tuple[_RealResolveFDDirectionSample, ...]
    nqsr_aux_phi: jax.Array
    nqsr_aux_theta: jax.Array


def _real_resolve_fd_iotas_gradient(booz_jax, bs_jax):
    del bs_jax
    return IotasJAX(booz_jax).dJ()


def _real_resolve_fd_iotas_value(real_resolve_fd_suite, outcome):
    del real_resolve_fd_suite
    assert outcome.iota is not None
    return float(outcome.iota)


def _real_resolve_fd_nqsr_gradient(booz_jax, bs_jax):
    return NonQuasiSymmetricRatioJAX(
        booz_jax,
        bs_jax,
        sDIM=_REAL_RESOLVE_FD_NQSR_SDIM,
    ).dJ()


def _real_resolve_fd_nqsr_value(real_resolve_fd_suite, outcome):
    assert outcome.coil_dofs is not None
    assert outcome.surface_dofs is not None
    bs_jax = real_resolve_fd_suite.bs_jax
    booz_jax = real_resolve_fd_suite.booz_jax
    coil_set_spec = bs_jax.coil_set_spec_from_dofs(
        jnp.asarray(outcome.coil_dofs, dtype=jnp.float64)
    )
    objective_value = _qs_ratio_pure(
        jnp.asarray(outcome.surface_dofs, dtype=jnp.float64),
        coil_set_spec,
        quadpoints_phi=real_resolve_fd_suite.nqsr_aux_phi,
        quadpoints_theta=real_resolve_fd_suite.nqsr_aux_theta,
        mpol=booz_jax.mpol,
        ntor=booz_jax.ntor,
        nfp=booz_jax.nfp,
        stellsym=booz_jax.stellsym,
        scatter_indices=booz_jax.scatter_indices,
        surface_kind=booz_jax._surface_geometry_kind,
        axis=0,
    )
    return float(np.asarray(jax.device_get(objective_value)))


def _real_resolve_fd_boozer_residual_gradient(booz_jax, bs_jax):
    return BoozerResidualJAX(booz_jax, bs_jax).dJ()


def _real_resolve_fd_boozer_residual_value(real_resolve_fd_suite, outcome):
    assert outcome.iota is not None
    assert outcome.coil_dofs is not None
    assert outcome.surface_dofs is not None
    bs_jax = real_resolve_fd_suite.bs_jax
    booz_jax = real_resolve_fd_suite.booz_jax
    optimize_G = outcome.G is not None
    coil_set_spec = bs_jax.coil_set_spec_from_dofs(
        jnp.asarray(outcome.coil_dofs, dtype=jnp.float64)
    )
    x_inner = booz_jax._pack_decision_vector(
        float(outcome.iota),
        None if outcome.G is None else float(outcome.G),
        sdofs=jnp.asarray(outcome.surface_dofs, dtype=jnp.float64),
    )
    objective_value = _boozer_residual_J_of_x_inner(
        x_inner,
        coil_set_spec=coil_set_spec,
        quadpoints_phi=booz_jax.quadpoints_phi,
        quadpoints_theta=booz_jax.quadpoints_theta,
        mpol=booz_jax.mpol,
        ntor=booz_jax.ntor,
        nfp=booz_jax.nfp,
        stellsym=booz_jax.stellsym,
        scatter_indices=booz_jax.scatter_indices,
        surface_kind=booz_jax._surface_geometry_kind,
        optimize_G=optimize_G,
        weight_inv_modB=bool(outcome.weight_inv_modB),
        constraint_weight=booz_jax.constraint_weight,
        targetlabel=booz_jax.targetlabel,
        label_type=booz_jax.label_type,
        phi_idx=booz_jax.phi_idx,
    )
    return float(np.asarray(jax.device_get(objective_value)))


_REAL_RESOLVE_FD_WRAPPER_SPECS = (
    _RealResolveFDWrapperSpec(
        label="IotasJAX",
        gradient_builder=_real_resolve_fd_iotas_gradient,
        value_builder=_real_resolve_fd_iotas_value,
    ),
    _RealResolveFDWrapperSpec(
        label="NonQuasiSymmetricRatioJAX",
        gradient_builder=_real_resolve_fd_nqsr_gradient,
        value_builder=_real_resolve_fd_nqsr_value,
    ),
    _RealResolveFDWrapperSpec(
        label="BoozerResidualJAX",
        gradient_builder=_real_resolve_fd_boozer_residual_gradient,
        value_builder=_real_resolve_fd_boozer_residual_value,
    ),
)
_REAL_RESOLVE_FD_WRAPPER_SPECS_BY_LABEL = {
    spec.label: spec for spec in _REAL_RESOLVE_FD_WRAPPER_SPECS
}


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
        _RealResolveFDBaselineState(
            coil_dofs=np.asarray(bs_jax.x, dtype=float).copy(),
            surface_dofs=np.asarray(booz_jax.surface.get_dofs(), dtype=float).copy(),
            iota=float(result["iota"]),
            G=float(result["G"]),
            fun=float(summarize_result_fun(result)),
            result=result,
        ),
    )


def _is_stable_real_resolve(baseline_state, *, iota_value, G_value, fun_value):
    return (
        abs(iota_value - baseline_state.iota) < _STABLE_IOTA_ABS_TOL
        and relative_error(G_value, baseline_state.G) < _STABLE_G_REL_TOL
        and relative_error(fun_value, baseline_state.fun) < _STABLE_FUN_REL_TOL
    )


def _unique_normalized_real_resolve_fd_directions(candidate_directions):
    directions = []
    for direction in candidate_directions:
        normalized_direction = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(normalized_direction)
        assert norm > 0.0
        normalized_direction = normalized_direction / norm
        if any(
            np.allclose(normalized_direction, existing_direction, rtol=0.0, atol=1e-12)
            or np.allclose(
                normalized_direction,
                -existing_direction,
                rtol=0.0,
                atol=1e-12,
            )
            for existing_direction in directions
        ):
            continue
        directions.append(normalized_direction)
    return tuple(directions)


def _real_resolve_fd_probe_directions(dof_count):
    """Return the canonical deterministic probe family for re-solve FD checks."""
    assert dof_count > 0
    last_index = dof_count - 1
    candidate_directions = []
    for fraction in _REAL_RESOLVE_FD_AXIS_DIRECTION_FRACTIONS:
        dof_index = int(round(fraction * last_index))
        direction = np.zeros(dof_count, dtype=float)
        direction[dof_index] = 1.0
        candidate_directions.append(direction)

    # Add deterministic mixed probes so the fixed family covers both axis-local
    # and coupled coil motion without relying on pseudo-random directions.
    candidate_directions.append(np.ones(dof_count, dtype=float))
    if dof_count > 1:
        alternating = np.ones(dof_count, dtype=float)
        alternating[1::2] = -1.0
        candidate_directions.append(alternating)

    return _unique_normalized_real_resolve_fd_directions(candidate_directions)


def _restore_real_resolve_fd_seed_state(baseline_state, bs_jax, booz_jax):
    bs_jax.x = np.asarray(baseline_state.coil_dofs, dtype=float)
    booz_jax.surface.set_dofs(np.asarray(baseline_state.surface_dofs, dtype=float))


def _resolve_real_fixture_probe_outcome(baseline_state, bs_jax, booz_jax, coil_dofs):
    # Reset to base seed state before applying the perturbation
    _restore_real_resolve_fd_seed_state(baseline_state, bs_jax, booz_jax)
    bs_jax.x = np.asarray(coil_dofs, dtype=float)
    booz_jax.need_to_run_code = True

    # Re-solve from the warm-start guess
    result = booz_jax.run_code(
        iota=baseline_state.iota,
        G=baseline_state.G,
    )

    if result is None or not result.get("success", False):
        return _RealResolveFDProbeOutcome(
            stable=False,
            reason="solve_failed",
            coil_dofs=None,
            surface_dofs=None,
            iota=None,
            G=None,
            weight_inv_modB=None,
        )

    is_self_intersecting, check_available = (
        single_stage_example.evaluate_surface_self_intersection(booz_jax.surface)
    )
    if check_available and is_self_intersecting:
        return _RealResolveFDProbeOutcome(
            stable=False,
            reason="self_intersecting",
            coil_dofs=None,
            surface_dofs=None,
            iota=None,
            G=None,
            weight_inv_modB=None,
        )

    iota_value = float(result["iota"])
    G_value = float(result["G"])
    fun_value = float(summarize_result_fun(result))
    if not _is_stable_real_resolve(
        baseline_state,
        iota_value=iota_value,
        G_value=G_value,
        fun_value=fun_value,
    ):
        return _RealResolveFDProbeOutcome(
            stable=False,
            reason="branch_switch",
            coil_dofs=None,
            surface_dofs=None,
            iota=iota_value,
            G=G_value,
            weight_inv_modB=None,
        )

    return _RealResolveFDProbeOutcome(
        stable=True,
        reason="ok",
        coil_dofs=np.asarray(coil_dofs, dtype=float).copy(),
        surface_dofs=np.asarray(booz_jax.surface.get_dofs(), dtype=float).copy(),
        iota=iota_value,
        G=G_value,
        weight_inv_modB=bool(result.get("weight_inv_modB", True)),
    )


def _build_real_resolve_fd_suite():
    bs_jax, booz_jax, baseline_state = _make_real_resolve_fd_setup()
    gradients = {
        spec.label: np.asarray(spec.gradient_builder(booz_jax, bs_jax), dtype=float)
        for spec in _REAL_RESOLVE_FD_WRAPPER_SPECS
    }
    x0 = np.asarray(baseline_state.coil_dofs, dtype=float)
    directions = _real_resolve_fd_probe_directions(len(x0))
    direction_samples = []
    for direction in directions:
        epsilon_samples = []
        for eps in _REAL_RESOLVE_FD_EPSILONS:
            epsilon_samples.append(
                _RealResolveFDEpsilonSample(
                    eps=eps,
                    plus=_resolve_real_fixture_probe_outcome(
                        baseline_state,
                        bs_jax,
                        booz_jax,
                        x0 + eps * direction,
                    ),
                    minus=_resolve_real_fixture_probe_outcome(
                        baseline_state,
                        bs_jax,
                        booz_jax,
                        x0 - eps * direction,
                    ),
                )
            )
        direction_samples.append(
            _RealResolveFDDirectionSample(
                direction=direction,
                epsilon_samples=tuple(epsilon_samples),
            )
        )

    _restore_real_resolve_fd_seed_state(baseline_state, bs_jax, booz_jax)
    booz_jax.res = baseline_state.result
    booz_jax.need_to_run_code = False

    return _RealResolveFDSuite(
        bs_jax=bs_jax,
        booz_jax=booz_jax,
        baseline_state=baseline_state,
        gradients=gradients,
        direction_samples=tuple(direction_samples),
        nqsr_aux_phi=jnp.asarray(
            np.linspace(
                0.0,
                1.0 / float(booz_jax.nfp),
                2 * _REAL_RESOLVE_FD_NQSR_SDIM,
                endpoint=False,
            ),
            dtype=jnp.float64,
        ),
        nqsr_aux_theta=jnp.asarray(
            np.linspace(
                0.0,
                1.0,
                2 * _REAL_RESOLVE_FD_NQSR_SDIM,
                endpoint=False,
            ),
            dtype=jnp.float64,
        ),
    )


@pytest.fixture(scope="module")
def real_resolve_fd_suite():
    # The reduced real-fixture re-solves dominate this lane's runtime.
    # Build the perturbation dataset once and reuse it across all wrappers.
    return _build_real_resolve_fd_suite()


def _real_resolve_fd_wrapper_value(
    real_resolve_fd_suite: _RealResolveFDSuite,
    *,
    wrapper_spec,
    outcome: _RealResolveFDProbeOutcome,
):
    if not outcome.stable:
        raise AssertionError("Expected a stable reduced real-fixture outcome")
    return wrapper_spec.value_builder(real_resolve_fd_suite, outcome)


def _assert_wrapper_resolve_fd_matches_real_fixture(
    *,
    wrapper_label,
    real_resolve_fd_suite,
):
    wrapper_spec = _REAL_RESOLVE_FD_WRAPPER_SPECS_BY_LABEL[wrapper_label]
    gradient = np.asarray(
        real_resolve_fd_suite.gradients[wrapper_spec.label], dtype=float
    )
    num_directions = len(real_resolve_fd_suite.direction_samples)

    stable_samples = 0
    instability_reasons = []
    short_series_reasons = []
    mismatch_reasons = []

    # Probe an explicit direction set and require several stable passes so the
    # test cannot succeed by exiting after one easy pseudo-random direction.
    for sample_index, direction_sample in enumerate(
        real_resolve_fd_suite.direction_samples
    ):
        direction = direction_sample.direction
        directional_adjoint = float(np.dot(gradient, direction))
        err_old = None
        stable_eps_count = 0
        direction_ok = True

        for epsilon_sample in direction_sample.epsilon_samples:
            eps = epsilon_sample.eps
            plus = epsilon_sample.plus
            minus = epsilon_sample.minus
            if not plus.stable or not minus.stable:
                instability_reasons.append(
                    f"sample {sample_index} eps={eps:.1e}: "
                    f"plus={plus.reason} minus={minus.reason}"
                )
                continue

            directional_fd = (
                _real_resolve_fd_wrapper_value(
                    real_resolve_fd_suite,
                    wrapper_spec=wrapper_spec,
                    outcome=plus,
                )
                - _real_resolve_fd_wrapper_value(
                    real_resolve_fd_suite,
                    wrapper_spec=wrapper_spec,
                    outcome=minus,
                )
            ) / (2.0 * eps)
            abs_err = abs(directional_adjoint - directional_fd)
            stable_eps_count += 1
            logger.info(
                f"{wrapper_label} reduced-real FD[{sample_index}, eps={eps:.1e}]: "
                f"adjoint={directional_adjoint:.6e} fd={directional_fd:.6e} "
                f"abs={abs_err:.2e}"
            )
            if err_old is not None:
                threshold = max(
                    _REAL_RESOLVE_FD_ABS_TOL,
                    _REAL_RESOLVE_FD_TAYLOR_RATE * err_old,
                )
                if abs_err >= threshold:
                    mismatch_reasons.append(
                        f"sample {sample_index} eps={eps:.1e}: "
                        f"err={abs_err:.2e} threshold={threshold:.2e}"
                    )
                    direction_ok = False
                    break
            err_old = abs_err

        if direction_ok and stable_eps_count >= _REAL_RESOLVE_FD_MIN_STABLE_EPS:
            stable_samples += 1
        elif direction_ok:
            short_series_reasons.append(
                f"sample {sample_index} stable eps={stable_eps_count}/"
                f"{len(_REAL_RESOLVE_FD_EPSILONS)}"
            )

    diagnostics = []
    if mismatch_reasons:
        diagnostics.append("Taylor mismatches: " + "; ".join(mismatch_reasons))
    if stable_samples < _REAL_RESOLVE_FD_MIN_STABLE_SAMPLES:
        instability_detail = (
            "; ".join(instability_reasons) if instability_reasons else "none"
        )
        short_series_detail = (
            "; ".join(short_series_reasons) if short_series_reasons else "none"
        )
        diagnostics.append(
            f"stable directions={stable_samples}/{num_directions} "
            f"(required {_REAL_RESOLVE_FD_MIN_STABLE_SAMPLES}); "
            f"instabilities: {instability_detail}; "
            f"short stable series: {short_series_detail}"
        )
    if diagnostics:
        pytest.fail(
            f"{wrapper_label} reduced real-fixture FD probes failed: "
            + " | ".join(diagnostics)
        )


class TestRealResolveFDProbeDirections:
    def test_probe_family_is_explicit_and_deterministic(self):
        directions = _real_resolve_fd_probe_directions(7)

        expected = []
        for dof_index in (0, 3, 6):
            direction = np.zeros(7, dtype=float)
            direction[dof_index] = 1.0
            expected.append(direction)

        expected.append(np.ones(7, dtype=float) / np.sqrt(7.0))
        alternating = np.ones(7, dtype=float)
        alternating[1::2] = -1.0
        expected.append(alternating / np.sqrt(7.0))

        assert len(directions) == len(expected) == 5
        for actual_direction, expected_direction in zip(directions, expected):
            np.testing.assert_allclose(actual_direction, expected_direction)


class TestRealResolveFDSuite:
    @pytest.mark.slow
    def test_shared_suite_restores_baseline_state(self, real_resolve_fd_suite):
        baseline_state = real_resolve_fd_suite.baseline_state
        np.testing.assert_allclose(
            np.asarray(real_resolve_fd_suite.bs_jax.x, dtype=float),
            np.asarray(baseline_state.coil_dofs, dtype=float),
        )
        np.testing.assert_allclose(
            np.asarray(real_resolve_fd_suite.booz_jax.surface.get_dofs(), dtype=float),
            np.asarray(baseline_state.surface_dofs, dtype=float),
        )

        result = real_resolve_fd_suite.booz_jax.res
        assert result is not None and result.get("success", False)
        assert real_resolve_fd_suite.booz_jax.need_to_run_code is False
        assert float(result["iota"]) == pytest.approx(baseline_state.iota)
        assert float(result["G"]) == pytest.approx(baseline_state.G)
        assert float(summarize_result_fun(result)) == pytest.approx(baseline_state.fun)


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

    shared_ls_options = {
        "verbose": False,
        "bfgs_maxiter": 300,
        "bfgs_tol": 1e-8,
        "newton_maxiter": 20,
        "newton_tol": 1e-9,
    }
    cpu_ls_options = {
        **shared_ls_options,
        "weight_inv_modB": weight_inv_modB,
    }
    jax_ls_options = {
        **shared_ls_options,
        "optimizer_backend": optimizer_backend,
        "weight_inv_modB": weight_inv_modB,
    }
    booz_cpu = BoozerSurface(
        bs_cpu,
        surf_cpu,
        vol_cpu,
        vol_target,
        constraint_weight=constraint_weight,
        options=cpu_ls_options,
    )
    booz_jax = BoozerSurfaceJAX(
        bs_jax,
        surf_jax,
        vol_jax,
        vol_target,
        constraint_weight=constraint_weight,
        options=jax_ls_options,
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


def _copy_optional_array(value):
    if value is None:
        return None
    return np.asarray(value, dtype=float).copy()


def _snapshot_solver_state(booz):
    return {
        "need_to_run_code": booz.need_to_run_code,
        "run_code": booz.run_code,
        "res_ref": booz.res,
        "res": None if booz.res is None else dict(booz.res),
    }


def _snapshot_boozer_setup_state(setup):
    (_, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, _) = setup
    return {
        "surf_cpu_dofs": np.asarray(surf_cpu.get_dofs(), dtype=float).copy(),
        "surf_jax_dofs": np.asarray(surf_jax.get_dofs(), dtype=float).copy(),
        "bs_cpu_points": _copy_optional_array(bs_cpu.get_points_cart_ref()),
        "bs_jax_points": _copy_optional_array(bs_jax._points_jax),
        "bs_jax_points_version": bs_jax._points_version,
        "bs_jax_x": np.asarray(bs_jax.x, dtype=float).copy(),
        "booz_cpu": _snapshot_solver_state(booz_cpu),
        "booz_jax": _snapshot_solver_state(booz_jax),
    }


def _restore_boozer_result(obj, res_ref, res_snapshot):
    if res_ref is None or res_snapshot is None:
        obj.res = None
        return
    res_ref.clear()
    res_ref.update(res_snapshot)
    obj.res = res_ref


def _restore_solver_state(booz, state):
    booz.run_code = state["run_code"]
    _restore_boozer_result(
        booz,
        state["res_ref"],
        state["res"],
    )
    booz.need_to_run_code = state["need_to_run_code"]


def _restore_boozer_setup_state(setup, state):
    (_, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, _) = setup
    surf_cpu.set_dofs(state["surf_cpu_dofs"])
    surf_jax.set_dofs(state["surf_jax_dofs"])
    bs_jax.x = state["bs_jax_x"]
    if state["bs_cpu_points"] is not None:
        bs_cpu.set_points(state["bs_cpu_points"])
    bs_jax._points_jax = (
        None
        if state["bs_jax_points"] is None
        else jnp.asarray(state["bs_jax_points"], dtype=jnp.float64)
    )
    bs_jax._points_version = state["bs_jax_points_version"]
    _restore_solver_state(booz_cpu, state["booz_cpu"])
    _restore_solver_state(booz_jax, state["booz_jax"])


@pytest.fixture(scope="module")
def _boozer_setup_module():
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
    assert res_cpu.get("success", False), "CPU solver did not converge"
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


@pytest.fixture
def boozer_setup(_boozer_setup_module):
    """Function-scoped view of the shared Boozer setup with guaranteed restore."""
    state = _snapshot_boozer_setup_state(_boozer_setup_module)
    yield _boozer_setup_module
    _restore_boozer_setup_state(_boozer_setup_module, state)


def test_make_boozer_setup_propagates_weight_inv_modB_to_cpu_and_jax():
    """Helper options should keep CPU and JAX LS weighting in sync."""
    (_, _, _, _, _, booz_cpu, booz_jax, _, _, _) = _make_boozer_setup(
        constraint_weight=1.0,
        weight_inv_modB=False,
    )

    assert booz_cpu.options["weight_inv_modB"] is False
    assert booz_jax.options["weight_inv_modB"] is False


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

        logger.info(f"BoozerResidual J: cpu={j_cpu:.12e} jax={j_jax:.12e}")
        # Both should be small on the shared reduced fixture.
        assert j_jax < 1e-3, f"JAX BoozerResidual too large: {j_jax:.2e}"
        assert j_cpu < 1e-3, f"CPU BoozerResidual too large: {j_cpu:.2e}"

    def test_value_path_matches_residual_helper_not_penalty_objective(
        self,
        boozer_setup,
        monkeypatch,
    ):
        """BoozerResidualJAX value must use the CPU-parity residual helper."""
        from simsopt._core.derivative import Derivative
        import simsopt.geo.surfaceobjectives_jax as soj

        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup

        captured = {}
        coil_dofs = jnp.array(bs_jax.x.copy())
        coil_set_spec = bs_jax.coil_set_spec_from_dofs(coil_dofs)

        jr_jax = BoozerResidualJAX(booz_jax, bs_jax)

        original_direct_objective = jr_jax._direct_objective_of_coils

        def raise_unexpected_call(message):
            raise AssertionError(message)

        def recording_direct_objective(coil_dofs, *objective_args):
            value = original_direct_objective(coil_dofs, *objective_args)
            captured["value"] = float(value)
            return value

        monkeypatch.setattr(soj, "_ensure_solved", lambda _booz: None)
        monkeypatch.setattr(
            soj,
            "_value_and_direct_coil_derivative",
            lambda *args, **kwargs: raise_unexpected_call(
                "BoozerResidualJAX.J() must not request the direct coil gradient"
            ),
        )
        monkeypatch.setattr(
            soj, "_solve_boozer_adjoint", lambda booz_surf, dJ_ds: dJ_ds
        )
        monkeypatch.setattr(
            soj, "_adjoint_coil_derivative", lambda *args, **kwargs: Derivative({})
        )
        monkeypatch.setattr(
            soj,
            "_boozer_penalty_objective",
            lambda *args, **kwargs: raise_unexpected_call(
                "BoozerResidualJAX.compute() must not route through "
                "_boozer_penalty_objective()"
            ),
            raising=False,
        )
        monkeypatch.setattr(
            jr_jax,
            "_direct_objective_of_coils",
            recording_direct_objective,
        )

        value = jr_jax.J()

        x_inner = booz_jax._pack_decision_vector(
            booz_jax.res["iota"],
            booz_jax.res["G"],
            sdofs=booz_jax._get_surface_dofs(),
        )

        expected = float(
            soj._boozer_residual_J_of_x_inner(
                x_inner,
                coil_set_spec=coil_set_spec,
                quadpoints_phi=booz_jax.quadpoints_phi,
                quadpoints_theta=booz_jax.quadpoints_theta,
                mpol=booz_jax.mpol,
                ntor=booz_jax.ntor,
                nfp=booz_jax.nfp,
                stellsym=booz_jax.stellsym,
                scatter_indices=booz_jax.scatter_indices,
                surface_kind=booz_jax._surface_geometry_kind,
                optimize_G=booz_jax.res["G"] is not None,
                weight_inv_modB=booz_jax.res.get("weight_inv_modB", True),
                constraint_weight=jr_jax.constraint_weight,
                targetlabel=booz_jax.targetlabel,
                label_type=booz_jax.label_type,
                phi_idx=booz_jax.phi_idx,
            )
        )

        np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)
        assert captured["value"] == value

    def test_direct_objective_value_and_grad_is_cached_per_instance(
        self,
        boozer_setup,
        monkeypatch,
    ):
        """BoozerResidualJAX should not rebuild its direct objective transform."""
        import simsopt.geo.surfaceobjectives_jax as soj

        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        original_factory = soj._make_cached_strict_scalar_value_and_grad
        build_count = 0

        def counting_factory(*args, **kwargs):
            nonlocal build_count
            build_count += 1
            return original_factory(*args, **kwargs)

        monkeypatch.setattr(
            soj,
            "_make_cached_strict_scalar_value_and_grad",
            counting_factory,
        )

        jr_jax = BoozerResidualJAX(booz_jax, bs_jax)
        cached_transform = jr_jax._direct_objective_value_and_grad

        assert build_count == 1
        jr_jax.J()
        jr_jax.recompute_bell()
        jr_jax.J()
        assert build_count == 1
        assert jr_jax._direct_objective_value_and_grad is cached_transform

    def test_constraint_weight_is_concrete_float_for_ls_surface(self, boozer_setup):
        """LS-only BoozerResidualJAX should store a concrete penalty weight."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup

        jr_jax = BoozerResidualJAX(booz_jax, bs_jax)

        assert isinstance(jr_jax.constraint_weight, float)
        assert jr_jax.constraint_weight == pytest.approx(
            float(booz_jax.constraint_weight)
        )


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

        logger.info(f"Iotas J: cpu={j_cpu:.12e} jax={j_jax:.12e}")
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
        logger.info(f"Adjoint residual: ||H^T adj - dJ_ds|| / ||dJ_ds|| = {rel:.2e}")
        assert rel < 1e-10, f"Adjoint solve residual too large: {rel:.2e}"

    def test_device_native_batched_adjoint_solve_matches_host(self, boozer_setup):
        """JAX transposed PLU solve should match host for matrix right-hand sides."""
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )
        from simsopt.objectives.utilities import forward_backward, forward_backward_jax

        P, L, U = booz_jax.res["PLU"]
        iota_rhs = _iota_unit_rhs((P, L, U))
        rhs_matrix = np.stack(
            (
                iota_rhs,
                2.0 * iota_rhs,
                -0.5 * iota_rhs,
            ),
            axis=1,
        )

        adj_host = forward_backward(P, L, U, rhs_matrix)
        adj_jax = np.asarray(forward_backward_jax(P, L, U, rhs_matrix))

        np.testing.assert_allclose(adj_jax, adj_host, rtol=1e-12, atol=1e-12)

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

        logger.info(f"||VJP result|| = {np.linalg.norm(g):.6e}")
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
        coils = [_FallbackBombCoil(), _FallbackBombCoil()]
        bs_jax = object.__new__(BiotSavartJAX)
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
        assert len(coils[0].current.calls) == 1
        assert len(coils[1].current.calls) == 1
        np.testing.assert_allclose(coils[0].current.calls[0], np.array([1.5]))
        np.testing.assert_allclose(coils[1].current.calls[0], np.array([2.5]))

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

    def test_coil_set_spec_from_dofs_prefers_immutable_coil_specs_when_available(
        self,
        monkeypatch,
    ):
        """Spec-capable curves should not rebuild grouped arrays on the adapter path."""
        curve = CurveXYZFourier(16, 1)
        curve.x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        current = Current(1.23)
        bs_jax = BiotSavartJAX([Coil(curve, current)])

        monkeypatch.setattr(
            bs_jax,
            "_coil_arrays_in_order_from_dofs",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError(
                    "coil_set_spec_from_dofs() should use immutable per-coil specs "
                    "before the grouped-array compatibility path"
                )
            ),
        )

        coil_set_spec = bs_jax.coil_set_spec_from_dofs(jnp.asarray(bs_jax.x))

        assert isinstance(coil_set_spec, GroupedCoilSetSpec)
        np.testing.assert_allclose(
            np.asarray(coil_set_spec.groups[0].gammas[0]),
            curve.gamma(),
            atol=1e-12,
        )

    def test_coil_specs_from_dofs_uses_cached_pytree_extraction_spec(
        self,
        monkeypatch,
    ):
        """Explicit coil reconstruction should stay off the live curve graph after init."""
        import simsopt.field.biotsavart_jax_backend as bsj

        curve = CurveXYZFourier(16, 1)
        curve.x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        current = Current(1.23)
        bs_jax = BiotSavartJAX([Coil(curve, current)])

        extraction_spec = bs_jax.coil_dof_extraction_spec()
        assert isinstance(extraction_spec, CoilSetDofExtractionSpec)

        monkeypatch.setattr(
            bsj,
            "curve_spec_from_curve",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError(
                    "coil_specs_from_dofs() should use the cached immutable "
                    "extraction spec after adapter initialization"
                )
            ),
        )

        coil_specs = bs_jax.coil_specs_from_dofs(jnp.asarray(bs_jax.x))

        assert len(coil_specs) == 1
        np.testing.assert_allclose(
            np.asarray(coil_specs[0].current.value),
            np.array([1.23]),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(coil_specs[0].curve.dofs),
            np.asarray(curve.full_x),
            atol=1e-12,
        )

    def test_strict_mode_allows_native_spec_reconstruction_in_coil_set_spec_from_dofs(
        self,
        monkeypatch,
        request,
    ):
        """Strict JAX mode should allow native immutable-spec reconstruction."""
        curve = _build_helical_curve(16)
        current = Current(1.23)
        bs_jax = BiotSavartJAX([Coil(curve, current)])
        _enable_strict_jax_backend(monkeypatch, request)
        grouped_spec = bs_jax.coil_set_spec_from_dofs(jnp.asarray(bs_jax.x))
        assert isinstance(grouped_spec, GroupedCoilSetSpec)

    def test_strict_mode_allows_native_spec_reconstruction_in_coil_set_spec(
        self,
        monkeypatch,
        request,
    ):
        """Strict JAX mode should allow the live native grouped-coil spec path."""
        curve = _build_helical_curve(16)
        current = Current(1.23)
        bs_jax = BiotSavartJAX([Coil(curve, current)])
        _enable_strict_jax_backend(monkeypatch, request)
        grouped_spec = bs_jax.coil_set_spec()
        assert isinstance(grouped_spec, GroupedCoilSetSpec)

    def test_non_strict_mode_warns_on_grouped_spec_fallback_in_coil_set_spec_from_dofs(
        self,
        monkeypatch,
        request,
    ):
        curve = CurveXYZFourier(16, 1)
        curve.x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        current = Current(1.23)
        bs_jax = BiotSavartJAX([Coil(curve, current)])
        monkeypatch.setattr(
            bs_jax,
            "coil_specs_from_dofs",
            lambda _coil_dofs: (_ for _ in ()).throw(NotImplementedError),
        )
        _assert_hidden_spec_fallback_warns(
            monkeypatch,
            request,
            lambda: bs_jax.coil_set_spec_from_dofs(jnp.asarray(bs_jax.x)),
            api_name="coil_set_spec_from_dofs",
        )

    @pytest.mark.parametrize("mode", ["strict", "non_strict"])
    def test_coil_set_spec_rejects_removed_live_graph_spec_seam(
        self,
        monkeypatch,
        request,
        mode,
    ):
        curve = CurveXYZFourier(16, 1)
        curve.x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        current = Current(1.23)
        bs_jax = BiotSavartJAX([Coil(curve, current)])
        monkeypatch.setattr(
            bs_jax,
            "_coil_set_spec_from_dofs_prefer_specs",
            lambda _coil_dofs: (_ for _ in ()).throw(NotImplementedError),
        )
        monkeypatch.setattr(
            bs_jax,
            "coil_specs",
            lambda: (_ for _ in ()).throw(NotImplementedError),
        )
        _assert_removed_live_graph_spec_seam(
            monkeypatch,
            request,
            bs_jax.coil_set_spec,
            mode=mode,
        )

    def test_extract_coil_data_grouped_uses_explicit_live_graph_compatibility_path(
        self,
        monkeypatch,
    ):
        curve = CurveXYZFourier(16, 1)
        curve.x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        current = Current(1.23)
        bs_jax = BiotSavartJAX([Coil(curve, current)])
        monkeypatch.setattr(
            bs_jax,
            "_coil_set_spec_from_dofs_prefer_specs",
            lambda _coil_dofs: (_ for _ in ()).throw(NotImplementedError),
        )
        monkeypatch.setattr(
            bs_jax,
            "coil_specs",
            lambda: (_ for _ in ()).throw(NotImplementedError),
        )
        expected = bs_jax._coil_set_spec_from_live_geometry()
        observed = bs_jax._extract_coil_data_grouped()
        _assert_grouped_field_data_matches_spec(observed, expected)

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

    def test_set_points_promotes_float32_inputs_to_float64(self):
        curve = CurveXYZFourier(16, 1)
        current = Current(1.23)
        bs_jax = BiotSavartJAX([Coil(curve, current)])
        points = np.asarray(
            [
                [0.1, 0.0, 0.0],
                [0.2, 0.1, -0.1],
            ],
            dtype=np.float32,
        )

        bs_jax.set_points(points)

        assert bs_jax._points_jax.dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(bs_jax._points_jax), points, atol=0.0)

    def test_curvexyzfourier_spec_pullback_matches_curve_methods(self):
        curve = CurveXYZFourier(16, 1)
        curve.set_dofs(np.array([1.0, 0.2, -0.1, 0.1, -0.05, 0.03, 0.8, -0.2, 0.15]))
        _assert_curve_spec_pullback_matches_curve_methods(curve)

    def test_curverzfourier_spec_pullback_matches_curve_methods(self):
        _assert_curve_spec_pullback_matches_curve_methods(_build_rz_curve(24))

    def test_curvehelical_spec_pullback_matches_curve_methods(self):
        _assert_curve_spec_pullback_matches_curve_methods(_build_helical_curve(24))

    def test_curveplanarfourier_spec_pullback_matches_curve_methods(self):
        _assert_curve_spec_pullback_matches_curve_methods(_build_planar_curve(24))

    def test_curveperturbed_spec_pullback_matches_curve_methods(self):
        _assert_curve_spec_pullback_matches_curve_methods(
            _build_perturbed_helical_curve(24)
        )

    def test_curvefilament_spec_pullback_matches_curve_methods(self):
        _assert_curve_spec_pullback_matches_curve_methods(_build_filament_curve(32))

    def test_curvecwsfouriercpp_spec_pullback_matches_curve_and_surface_methods(self):
        curve, _surf = _build_surface_bound_cpp_curve(32)
        _assert_curve_spec_pullback_matches_curve_methods(
            curve,
            expect_surface=True,
        )

    def test_curvecwsfourier_spec_pullback_matches_curve_and_surface_methods(self):
        curve, _surf = _build_surface_bound_jax_curve(32)
        _assert_curve_spec_pullback_matches_curve_methods(
            curve,
            expect_surface=True,
        )

    def test_curvehelical_exposes_immutable_spec(self):
        _assert_curve_exposes_immutable_spec(_build_helical_curve(24), CurveHelicalSpec)

    def test_curveplanarfourier_exposes_immutable_spec(self):
        _assert_curve_exposes_immutable_spec(
            _build_planar_curve(24),
            CurvePlanarFourierSpec,
        )

    def test_curveperturbed_exposes_immutable_spec(self):
        _assert_curve_exposes_immutable_spec(
            _build_perturbed_helical_curve(24),
            CurvePerturbedSpec,
        )

    def test_curveperturbed_to_spec_preserves_sample_derivatives(self):
        curve = _build_perturbed_helical_curve(24)
        spec = curve.to_spec()

        np.testing.assert_allclose(np.asarray(spec.sample_gamma), curve.sample[0])
        np.testing.assert_allclose(np.asarray(spec.sample_gammadash), curve.sample[1])
        np.testing.assert_allclose(
            np.asarray(spec.sample_gammadashdash),
            curve.sample[2],
        )
        np.testing.assert_allclose(
            np.asarray(spec.sample_gammadashdashdash),
            curve.sample[3],
        )

    def test_curvefilament_exposes_immutable_spec(self):
        _assert_curve_exposes_immutable_spec(
            _build_filament_curve(32),
            CurveFilamentSpec,
        )

    def test_curvefilament_to_spec_preserves_frame_and_offsets(self):
        curve = _build_filament_curve(32)
        spec = curve.to_spec()

        assert spec.frame_kind == "centroid"
        assert spec.dn == pytest.approx(curve.dn)
        assert spec.db == pytest.approx(curve.db)
        np.testing.assert_allclose(
            np.asarray(spec.rotation.dofs),
            curve.rotation.full_x,
            atol=1e-12,
        )

    @pytest.mark.parametrize(
        ("curve_builder", "nquad"),
        (
            (_build_xyz_curve, 32),
            (_build_perturbed_helical_curve, 24),
            (_build_filament_curve, 32),
        ),
        ids=("curvexyzfourier", "curveperturbed", "curvefilament"),
    )
    def test_curve_geometry_from_dofs_matches_live_curve(self, curve_builder, nquad):
        _assert_curve_spec_geometry_matches_live_curve(curve_builder(nquad))

    @pytest.mark.parametrize(
        ("curve_builder", "nquad"),
        (
            (_build_xyz_curve, 32),
            (_build_perturbed_helical_curve, 24),
            (_build_filament_curve, 32),
        ),
        ids=("curvexyzfourier", "curveperturbed", "curvefilament"),
    )
    def test_curve_gamma_and_dash_gradient_matches_fd(self, curve_builder, nquad):
        _assert_curve_spec_gamma_and_dash_gradient_matches_fd(curve_builder(nquad))

    def test_optimizable_dof_map_scatter_matches_expected_segments(self):
        map_spec = make_optimizable_dof_map_spec(
            template_full_dofs=jnp.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
            owner_segments=((0, 2, 1, 3), (2, 4, 4, 6)),
            input_mode="local",
            input_start=1,
            input_end=5,
        )
        owner_dofs = jnp.asarray([1.0, 2.0, 3.0, 4.0])

        mapped_full = _mapped_full_dofs(map_spec, owner_dofs)
        mapped_input = _mapped_input_dofs(map_spec, owner_dofs)

        np.testing.assert_allclose(
            np.asarray(mapped_full),
            np.asarray([10.0, 1.0, 2.0, 40.0, 3.0, 4.0]),
        )
        np.testing.assert_allclose(
            np.asarray(mapped_input),
            np.asarray([1.0, 2.0, 40.0, 3.0]),
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

    def test_coil_set_spec_from_dofs_prefers_immutable_curvehelical_specs(
        self, monkeypatch
    ):
        _assert_coil_set_spec_prefers_immutable_curve_specs(
            monkeypatch,
            _build_helical_curve(24),
            1.23,
            "CurveHelical should use immutable specs before grouped-array fallback",
        )

    def test_coil_set_spec_from_dofs_prefers_immutable_curveplanarfourier_specs(
        self,
        monkeypatch,
    ):
        _assert_coil_set_spec_prefers_immutable_curve_specs(
            monkeypatch,
            _build_planar_curve(24),
            2.5,
            "CurvePlanarFourier should use immutable specs before grouped-array fallback",
        )

    def test_grouped_coil_arrays_from_dofs_supports_curveperturbed_fullgraph_geometry(
        self,
    ):
        """Full-graph wrapper curves should reconstruct from explicit coil DOFs."""
        _assert_grouped_coil_arrays_match_curve(_build_perturbed_helical_curve(24), 1.7)

    def test_coil_set_spec_from_dofs_prefers_immutable_curveperturbed_specs(
        self,
        monkeypatch,
    ):
        _assert_coil_set_spec_prefers_immutable_curve_specs(
            monkeypatch,
            _build_perturbed_helical_curve(24),
            1.7,
            "CurvePerturbed should use immutable specs before grouped-array fallback",
        )

    def test_coil_set_spec_from_dofs_prefers_immutable_curvefilament_specs(
        self,
        monkeypatch,
    ):
        _assert_coil_set_spec_prefers_immutable_curve_specs(
            monkeypatch,
            _build_filament_curve(32),
            8.0e4,
            "CurveFilament should use immutable specs before grouped-array fallback",
        )

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

    def test_non_strict_mode_warns_on_public_cpu_coil_vjp_pullback(
        self,
        monkeypatch,
        request,
    ):
        """Non-strict JAX mode should warn before using the public CPU pullback seam."""
        _enable_non_strict_jax_backend(monkeypatch, request)
        coils = [_CpuFallbackRecordingCoil(), _CpuFallbackRecordingCoil()]
        bs_jax = _make_biotsavart_jax_for_coils(coils)
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

        with pytest.warns(
            RuntimeWarning,
            match=_PUBLIC_COIL_VJP_WARNING_PATTERN,
        ):
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
        assert len(coils[0].calls) == 1
        assert len(coils[1].calls) == 1
        assert len(coils[0].current.calls) == 1
        assert len(coils[1].current.calls) == 1

    def test_strict_mode_rejects_public_cpu_coil_vjp_pullback(
        self,
        monkeypatch,
        request,
    ):
        """Strict JAX mode should reject the public CPU pullback seam outright."""
        _enable_strict_jax_backend(monkeypatch, request)
        coils = [_CpuFallbackRecordingCoil()]
        bs_jax = _make_biotsavart_jax_for_coils(coils)

        with pytest.raises(
            RuntimeError,
            match=_PUBLIC_COIL_VJP_STRICT_PATTERN,
        ):
            bs_jax.coil_cotangents_to_derivative(
                _single_coil_cotangent_arrays(
                    np.array([1.0, 2.0, 3.0]),
                    np.array([4.0, 5.0, 6.0]),
                    1.5,
                ),
                [[0]],
            )

        assert coils[0].calls == []
        assert coils[0].current.calls == []

    def test_fast_mode_rejects_public_cpu_coil_vjp_pullback(
        self,
        monkeypatch,
        request,
    ):
        """The fast/ondevice lane must not silently route through ``coil.vjp()``."""
        _enable_fast_non_strict_jax_backend(monkeypatch, request)
        coils = [_CpuFallbackRecordingCoil()]
        bs_jax = _make_biotsavart_jax_for_coils(coils)

        with pytest.raises(
            RuntimeError,
            match="BiotSavartJAX.*public CPU coil\\.vjp\\(\\) pullback compatibility path.*jax_gpu_fast.*fast/ondevice lane",
        ):
            bs_jax.coil_cotangents_to_derivative(
                _single_coil_cotangent_arrays(
                    np.array([1.0, 2.0, 3.0]),
                    np.array([4.0, 5.0, 6.0]),
                    1.5,
                ),
                [[0]],
            )

        assert coils[0].calls == []
        assert coils[0].current.calls == []

    def test_biotsavart_projection_preserves_raw_rotated_cpu_coil_vjp_inputs(
        self,
        monkeypatch,
        request,
    ):
        """Rotated public fallback coils should receive raw cotangents at ``coil.vjp()``."""
        _enable_non_strict_jax_backend(monkeypatch, request)
        coils = [_CpuFallbackRecordingCoil(rotated=True, phi=np.pi / 2.0)]
        bs_jax = _make_biotsavart_jax_for_coils(coils)
        d_gamma = np.array([1.0, 2.0, 3.0])
        d_gammadash = np.array([4.0, 5.0, 6.0])

        with pytest.warns(
            RuntimeWarning,
            match=_PUBLIC_COIL_VJP_WARNING_PATTERN,
        ):
            derivative = bs_jax.coil_cotangents_to_derivative(
                _single_coil_cotangent_arrays(d_gamma, d_gammadash, 1.5),
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
        np.testing.assert_allclose(coils[0].calls[0][0], d_gamma, atol=1e-12)
        np.testing.assert_allclose(coils[0].calls[0][1], d_gammadash, atol=1e-12)
        np.testing.assert_allclose(coils[0].calls[0][2], np.array([1.5]), atol=1e-12)
        assert len(coils[0].calls) == 1
        assert len(coils[0].current.calls) == 1

    def test_biotsavart_projection_rejects_unsupported_curves_without_coil_fallback(
        self,
    ):
        """Unsupported curves should fail fast instead of falling back to ``coil.vjp()``."""
        coils = [_RecordingVJPCoil()]
        bs_jax = _make_biotsavart_jax_for_coils(coils)

        with pytest.raises(TypeError, match="supported JAX or CPU pullback contract"):
            bs_jax.coil_cotangents_to_derivative(
                _single_coil_cotangent_arrays(
                    np.array([1.0, 2.0, 3.0]),
                    np.array([4.0, 5.0, 6.0]),
                    1.5,
                ),
                [[0]],
            )

        assert coils[0].calls == []

    def test_biotsavart_projection_rejects_rotated_unsupported_curves_without_coil_fallback(
        self,
    ):
        """Rotated wrappers must not make unsupported base curves look pullback-capable."""
        coils = [_RotatedUnsupportedRecordingCoil()]
        bs_jax = _make_biotsavart_jax_for_coils(coils)

        with pytest.raises(TypeError, match="supported JAX or CPU pullback contract"):
            bs_jax.coil_cotangents_to_derivative(
                _single_coil_cotangent_arrays(
                    np.array([1.0, 2.0, 3.0]),
                    np.array([4.0, 5.0, 6.0]),
                    1.5,
                ),
                [[0]],
            )

        assert coils[0].calls == []
        assert coils[0].current.calls == []

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

    def test_coil_cotangent_projection_uses_jax_path_for_projectable_curves(self):
        """BiotSavartJAX should project directly through the shared JAX path."""
        coils = [_FallbackBombCoil()]
        bs_jax = object.__new__(BiotSavartJAX)
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

    def test_legacy_surfaceobjectives_projection_helper_is_unsupported(self):
        """The old surfaceobjectives compatibility helper should hard-fail now."""
        from simsopt.geo.surfaceobjectives_jax import _coil_cotangents_to_derivative

        with pytest.raises(
            RuntimeError,
            match="BiotSavartJAX\\.coil_cotangents_to_derivative",
        ):
            _coil_cotangents_to_derivative(
                [_RecordingVJPCoil()],
                [
                    (
                        jnp.array([[1.0, 2.0, 3.0]]),
                        jnp.array([[4.0, 5.0, 6.0]]),
                        jnp.array([1.5]),
                    )
                ],
                [[0]],
            )

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

        gradients = _jax_single_stage_wrapper_gradients(booz_jax, bs_jax)

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

        gradients = _jax_single_stage_wrapper_gradients(booz_jax, bs_jax)

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
        """Grouped LS VJPs should also match when ``optimize_G=False``.

        The fixed-``G`` LS lane can stop with a finite low-cost Newton exit
        while still producing the PLU/VJP state needed by the adjoint path.
        This test only requires that adjoint state, not a globally successful
        Newton polish flag.
        """
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
        assert res_ls["PLU"] is not None
        assert callable(res_ls["vjp"])
        assert callable(res_ls["vjp_groups"])

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

    @pytest.mark.parametrize(
        "objective_factory",
        (
            lambda booz_jax, bs_jax: BoozerResidualJAX(booz_jax, bs_jax),
            lambda booz_jax, bs_jax: IotasJAX(booz_jax),
            lambda booz_jax, bs_jax: NonQuasiSymmetricRatioJAX(
                booz_jax,
                bs_jax,
                sDIM=6,
            ),
        ),
    )
    def test_surface_objective_wrappers_reject_missing_streaming_group_vjp(
        self,
        boozer_setup,
        objective_factory,
    ):
        """Objective wrappers should hard-fail if the legacy full VJP seam reappears."""
        (coils, surf_cpu, surf_jax, bs_cpu, bs_jax, booz_cpu, booz_jax, vol_cpu) = (
            boozer_setup
        )
        booz_jax.res["vjp_groups"] = None

        with pytest.raises(RuntimeError, match="legacy full-pytree adjoint fallback"):
            objective_factory(booz_jax, bs_jax).dJ()

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

        logger.info(f"NonQSRatio J: cpu={j_cpu:.12e} jax={j_jax:.12e}")
        # Both must be finite and non-negative (solvers converge to different
        # surfaces, so exact parity is not expected)
        assert np.isfinite(j_jax) and j_jax >= 0, f"JAX NonQSRatio invalid: {j_jax}"
        assert np.isfinite(j_cpu) and j_cpu >= 0, f"CPU NonQSRatio invalid: {j_cpu}"

    def test_threads_surface_kind_into_qs_ratio(self, boozer_setup, monkeypatch):
        """The QS-ratio wrapper must pass the active surface geometry contract through."""
        from simsopt._core.derivative import Derivative
        import simsopt.geo.surfaceobjectives_jax as soj

        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        recorded_surface_kinds = []

        def fake_qs_ratio_pure(sdofs, coil_set_spec, **qs_kwargs):
            del coil_set_spec
            recorded_surface_kinds.append(qs_kwargs["surface_kind"])
            return jnp.sum(sdofs**2)

        monkeypatch.setattr(booz_jax, "_surface_geometry_kind", "rzfourier")
        monkeypatch.setattr(soj, "_qs_ratio_pure", fake_qs_ratio_pure)
        monkeypatch.setattr(
            soj, "_solve_boozer_adjoint", lambda booz_surf, dJ_ds: dJ_ds
        )
        monkeypatch.setattr(
            soj, "_adjoint_coil_derivative", lambda *args, **kwargs: Derivative({})
        )

        value = NonQuasiSymmetricRatioJAX(booz_jax, bs_jax, sDIM=6).J()

        assert np.isfinite(value)
        assert recorded_surface_kinds
        assert set(recorded_surface_kinds) == {"rzfourier"}

    def test_uses_spec_reconstruction_not_grouped_arrays(
        self,
        boozer_setup,
        monkeypatch,
    ):
        """NonQSRatioJAX must rebuild coil data through immutable specs."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup

        original_coil_set_spec_from_dofs = bs_jax.coil_set_spec_from_dofs
        calls = {"count": 0}

        def _counting_coil_set_spec_from_dofs(coil_dofs):
            calls["count"] += 1
            return original_coil_set_spec_from_dofs(coil_dofs)

        def _reject_grouped_arrays(*_args, **_kwargs):
            raise AssertionError(
                "NonQSRatioJAX should not call grouped_coil_arrays_from_dofs()"
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

        nqs_jax = NonQuasiSymmetricRatioJAX(booz_jax, bs_jax, sDIM=6)
        gradient = np.array(nqs_jax.dJ())

        assert np.all(np.isfinite(gradient))
        assert calls["count"] > 0

    def test_dj_allows_strict_transfer_guard(self, monkeypatch, request):
        """NonQSRatioJAX.dJ() must stay strict-safe on the direct coil path."""
        import simsopt.config as simsopt_config
        from simsopt.backend import invalidate_backend_cache

        monkeypatch.setenv("SIMSOPT_JAX_TRANSFER_GUARD", "disallow")
        enable_strict_jax_backend(monkeypatch, request, mode="jax_cpu_parity")
        invalidate_backend_cache()
        request.addfinalizer(invalidate_backend_cache)
        simsopt_config.set_backend(
            "jax_cpu_parity",
            strict=True,
            transfer_guard="disallow",
        )

        (
            _coils,
            _surf_cpu,
            _surf_jax,
            _bs_cpu,
            bs_jax,
            _booz_cpu,
            booz_jax,
            _vol_cpu,
            iota0,
            G0,
        ) = _make_boozer_setup(constraint_weight=1.0, optimizer_backend="ondevice")

        result = booz_jax.run_code(iota0, G0)
        assert result is not None and result.get("success", False)

        gradient = np.array(NonQuasiSymmetricRatioJAX(booz_jax, bs_jax, sDIM=6).dJ())

        assert gradient.size > 0
        assert np.all(np.isfinite(gradient))


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

        logger.info(f"Composite JAX: J={j:.12e} ||dJ||={np.linalg.norm(g):.6e}")
        assert np.isfinite(j), "Composite J is not finite"
        assert np.all(np.isfinite(g)), "Composite dJ contains NaN/inf"

    def test_boozer_residual_uses_spec_reconstruction_not_live_field_path(
        self,
        boozer_setup,
        monkeypatch,
    ):
        """BoozerResidualJAX should rebuild coil data through immutable specs."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup

        original_coil_set_spec_from_dofs = bs_jax.coil_set_spec_from_dofs
        calls = {"count": 0}

        def _counting_coil_set_spec_from_dofs(coil_dofs):
            calls["count"] += 1
            return original_coil_set_spec_from_dofs(coil_dofs)

        def _reject_set_points(*_args, **_kwargs):
            raise AssertionError(
                "BoozerResidualJAX should not call set_points() on the spec path"
            )

        def _reject_B_vjp(*_args, **_kwargs):
            raise AssertionError(
                "BoozerResidualJAX should not call B_vjp() on the spec path"
            )

        monkeypatch.setattr(
            bs_jax,
            "coil_set_spec_from_dofs",
            _counting_coil_set_spec_from_dofs,
        )
        monkeypatch.setattr(bs_jax, "set_points", _reject_set_points)
        monkeypatch.setattr(bs_jax, "B_vjp", _reject_B_vjp)

        jr_jax = BoozerResidualJAX(booz_jax, bs_jax)
        gradient = np.array(jr_jax.dJ())

        assert np.all(np.isfinite(gradient))
        assert calls["count"] > 0

    def test_batched_standard_wrapper_gradients_match_separate_wrapper_computes(
        self,
        boozer_setup,
        monkeypatch,
    ):
        """The standard wrapper helper should share one adjoint solve."""
        import simsopt.geo.surfaceobjectives_jax as soj

        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup

        reference_gradients = [
            np.asarray(BoozerResidualJAX(booz_jax, bs_jax).dJ(), dtype=float),
            np.asarray(IotasJAX(booz_jax).dJ(), dtype=float),
            np.asarray(
                NonQuasiSymmetricRatioJAX(booz_jax, bs_jax, sDIM=6).dJ(),
                dtype=float,
            ),
        ]

        boozer_residual, iotas, non_qs_ratio = _make_jax_standard_wrapper_triplet(
            booz_jax,
            bs_jax,
        )

        original_solve = soj._solve_boozer_adjoint
        recorded = {"calls": 0}

        def counting_solve(booz_surf, rhs):
            recorded["calls"] += 1
            recorded["rhs_shape"] = tuple(np.shape(rhs))
            return original_solve(booz_surf, rhs)

        monkeypatch.setattr(soj, "_solve_boozer_adjoint", counting_solve)

        returned_gradients = compute_standard_surface_objective_gradients(
            boozer_residual,
            iotas,
            non_qs_ratio,
        )
        batched_gradients = [
            np.asarray(gradient, dtype=float) for gradient in returned_gradients
        ]

        assert recorded["calls"] == 1
        assert recorded["rhs_shape"] == (booz_jax.res["PLU"][1].shape[0],)
        for batched_gradient, reference_gradient in zip(
            batched_gradients,
            reference_gradients,
        ):
            np.testing.assert_allclose(
                batched_gradient,
                reference_gradient,
                rtol=5e-4,
                atol=1e-6,
            )
        np.testing.assert_allclose(
            np.asarray(boozer_residual.dJ()), batched_gradients[0]
        )
        np.testing.assert_allclose(np.asarray(iotas.dJ()), batched_gradients[1])
        np.testing.assert_allclose(np.asarray(non_qs_ratio.dJ()), batched_gradients[2])


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
            logger.info(
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

        logger.info(f"Composite: J={j0:.6e}, ||dJ||={grad_norm:.6e}")

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

        spec.loader.exec_module(mod)

        fake_vol = MagicMock()
        fake_vol.return_value = type("Volume", (), {})()
        with patch(
            "simsopt.geo.boozersurface_jax.BoozerSurfaceJAX", recorder
        ), patch.object(mod, "Volume", fake_vol), patch.object(
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
        logger.info("initialize_boozer_surface(backend='jax') -> BoozerSurfaceJAX OK")

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

        logger.info(
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


class TestRealFixtureOndeviceM5Parity:
    """Reduced real single-stage fixture covers the public on-device M5 lane."""

    def test_real_fixture_ondevice_solver_end_state_contracts_match(self):
        """CPU reference and JAX on-device lanes should match on solved-state quality, not iterates."""
        cpu_fixture, jax_fixture = _build_real_fixture_ondevice_m5_pair()

        _assert_boozer_surfaces_end_state_parity(
            "CPU",
            cpu_fixture["boozer_surface"],
            "JAX CPU",
            jax_fixture["boozer_surface"],
            tolerances=_REAL_FIXTURE_SOLVER_CPU_JAX_TOLS,
        )

    def test_real_fixture_ondevice_parity_and_wrapper_gradients(self):
        """CPU reference and JAX reduced-real on-device fixtures agree, and wrappers stay healthy."""
        cpu_fixture, jax_fixture = _build_real_fixture_ondevice_m5_pair()

        booz_cpu = cpu_fixture["boozer_surface"]
        bs_cpu = cpu_fixture["bs"]
        booz_jax = jax_fixture["boozer_surface"]
        bs_jax = jax_fixture["bs"]

        assert cpu_fixture["boozer_optimizer_backend"] is None
        assert jax_fixture["boozer_optimizer_backend"] == "ondevice"
        assert booz_cpu.res is not None and booz_cpu.res.get("success", False)
        assert booz_jax.res is not None and booz_jax.res.get("success", False)
        assert booz_jax.res["type"] == "ls"

        cpu_values = _cpu_single_stage_wrapper_values(booz_cpu, bs_cpu)
        jax_values = _jax_single_stage_wrapper_values(booz_jax, bs_jax)
        residual_cpu, iota_cpu, nqs_cpu = cpu_values
        residual_jax, iota_jax, nqs_jax = jax_values

        np.testing.assert_allclose(
            booz_jax.res["iota"],
            booz_cpu.res["iota"],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            booz_jax.res["G"],
            booz_cpu.res["G"],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(iota_jax, iota_cpu, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            residual_jax,
            residual_cpu,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(nqs_jax, nqs_cpu, rtol=1e-12, atol=1e-12)

        cpu_gradients = _cpu_single_stage_wrapper_gradients(booz_cpu, bs_cpu)
        jax_gradients = _jax_single_stage_wrapper_gradients(booz_jax, bs_jax)
        _assert_gradients_finite_nonzero(
            cpu_gradients,
            "Real-fixture CPU-reference wrapper path",
        )
        _assert_gradients_finite_nonzero(
            jax_gradients,
            "Real-fixture JAX ondevice wrapper path",
        )

        for label, cpu_gradient, jax_gradient in zip(
            ("BoozerResidual", "Iotas", "NonQuasiSymmetricRatio"),
            cpu_gradients,
            jax_gradients,
        ):
            np.testing.assert_allclose(
                jax_gradient,
                cpu_gradient,
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"{label} gradient mismatch on reduced real ondevice fixture",
            )


class TestRealFixtureGpuM5Parity:
    """Reduced real single-stage fixture covers the public GPU-backed M5 lane."""

    @pytest.mark.slow
    def test_real_fixture_gpu_solver_stays_ondevice_under_disallow(
        self,
        monkeypatch,
        request,
    ):
        gpu = parity_device("gpu")
        monkeypatch.setenv("SIMSOPT_JAX_TRANSFER_GUARD", "disallow")
        _enable_strict_jax_backend(monkeypatch, request)
        booz_cpu, gpu_fixture, gpu_result = _build_real_fixture_gpu_solver_pair()
        booz_gpu = gpu_fixture["boozer_surface"]

        assert str(gpu_result["optimizer_method"]).endswith("-ondevice")
        _assert_gpu_boozer_solver_result_on_device(gpu, gpu_fixture, gpu_result)
        _assert_boozer_surfaces_end_state_parity(
            "CPU",
            booz_cpu,
            "JAX GPU disallow",
            booz_gpu,
            tolerances=_REAL_FIXTURE_SOLVER_CPU_GPU_TOLS,
        )

    @pytest.mark.slow
    def test_real_fixture_gpu_solver_end_state_contracts_match_cpu_reference(
        self,
        monkeypatch,
        request,
    ):
        gpu = parity_device("gpu")
        _enable_strict_jax_backend(monkeypatch, request)
        booz_cpu, gpu_fixture, gpu_result = _build_real_fixture_gpu_solver_pair()
        _assert_gpu_boozer_solver_result_on_device(gpu, gpu_fixture, gpu_result)

        _assert_boozer_surfaces_end_state_parity(
            "CPU",
            booz_cpu,
            "JAX GPU",
            gpu_fixture["boozer_surface"],
            tolerances=_REAL_FIXTURE_SOLVER_CPU_GPU_TOLS,
        )

    @pytest.mark.slow
    def test_real_fixture_gpu_wrapper_values_and_gradients_match_cpu_reference(
        self,
        monkeypatch,
        request,
    ):
        """GPU wrapper values/gradients should match the reduced real CPU reference."""
        gpu = parity_device("gpu")
        _enable_strict_jax_backend(monkeypatch, request)

        cpu_fixture = build_real_single_stage_init_fixture(
            backend="cpu",
            optimizer_backend="scipy",
        )
        booz_cpu = cpu_fixture["boozer_surface"]
        bs_cpu = cpu_fixture["bs"]
        cpu_result = booz_cpu.res

        assert cpu_result is not None and cpu_result.get("success", False)

        gpu_fixture = build_real_single_stage_init_fixture(
            backend="jax",
            optimizer_backend="ondevice",
            boozer_surface_dofs_override=np.asarray(
                booz_cpu.surface.get_dofs(),
                dtype=float,
            ),
            boozer_iota_override=float(cpu_result["iota"]),
            boozer_G_override=float(cpu_result["G"]),
        )
        booz_gpu = gpu_fixture["boozer_surface"]
        bs_gpu = gpu_fixture["bs"]
        gpu_result = booz_gpu.res

        assert gpu_fixture["boozer_optimizer_backend"] == "ondevice"
        assert gpu_result is not None and gpu_result.get("success", False)
        assert gpu_result["type"] == "ls"
        assert_arrays_on_device(
            gpu,
            gpu_result["jacobian"],
            gpu_result["hessian"],
            *gpu_result["PLU"],
        )

        cpu_values = _cpu_single_stage_wrapper_values(booz_cpu, bs_cpu)
        gpu_values = _jax_single_stage_wrapper_values(booz_gpu, bs_gpu)
        residual_cpu, iota_cpu, nqs_cpu = cpu_values
        residual_gpu, iota_gpu, nqs_gpu = gpu_values

        np.testing.assert_allclose(
            gpu_result["iota"],
            cpu_result["iota"],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            gpu_result["G"],
            cpu_result["G"],
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(iota_gpu, iota_cpu, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            residual_gpu,
            residual_cpu,
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(nqs_gpu, nqs_cpu, rtol=1e-10, atol=1e-12)

        cpu_gradients = _cpu_single_stage_wrapper_gradients(booz_cpu, bs_cpu)
        gpu_gradients = _jax_single_stage_wrapper_gradients(booz_gpu, bs_gpu)
        _assert_gradients_finite_nonzero(
            cpu_gradients,
            "Real-fixture CPU wrapper path",
        )
        _assert_gradients_finite_nonzero(
            gpu_gradients,
            "Real-fixture GPU JAX wrapper path",
        )

        for label, cpu_gradient, gpu_gradient in zip(
            ("BoozerResidual", "Iotas", "NonQuasiSymmetricRatio"),
            cpu_gradients,
            gpu_gradients,
        ):
            np.testing.assert_allclose(
                gpu_gradient,
                cpu_gradient,
                rtol=1e-8,
                atol=1e-10,
                err_msg=f"{label} GPU gradient mismatch on reduced real fixture",
            )


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

        logger.info(
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
        logger.info(
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
        not private_optimizer_runtime_is_supported(jax.__version__),
        reason=f"On-device backend integration requires JAX >= {PRIVATE_OPTIMIZER_JAX_VERSION}.",
    )
    @pytest.mark.parametrize("optimizer_backend", ["ondevice"])
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

    @pytest.mark.parametrize(
        "wrapper_name",
        ["BoozerResidual", "Iotas", "NonQuasiSymmetricRatio"],
    )
    def test_cpu_wrappers_defer_guarded_adjoint_failure_to_dJ(
        self, boozer_setup, wrapper_name
    ):
        """A guarded adjoint callback must not break the value path."""
        (_, _, _, bs_cpu, _, booz_cpu, _, _) = boozer_setup
        old_res = booz_cpu.res
        old_dirty = booz_cpu.need_to_run_code

        bad_res = dict(old_res)
        bad_res["vjp"] = _make_guarded_gradient_failure("BoozerSurface")
        booz_cpu.res = bad_res
        booz_cpu.need_to_run_code = False
        try:
            obj = _build_boozer_wrapper(wrapper_name, booz_cpu, bs_cpu)
            assert np.isfinite(float(obj.J()))
            with pytest.raises(
                ValueError, match="requires fixed coil currents when G=None"
            ):
                obj.dJ()
        finally:
            booz_cpu.res = old_res
            booz_cpu.need_to_run_code = old_dirty

    @pytest.mark.parametrize(
        "wrapper_name",
        ["BoozerResidualJAX", "IotasJAX", "NonQuasiSymmetricRatioJAX"],
    )
    def test_jax_wrappers_defer_guarded_adjoint_failure_to_dJ(
        self, boozer_setup, wrapper_name
    ):
        """A guarded grouped-adjoint callback must not break the value path."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        old_res = booz_jax.res
        old_dirty = booz_jax.need_to_run_code

        bad_res = dict(old_res)
        bad_res["vjp_groups"] = _make_guarded_gradient_failure("BoozerSurfaceJAX")
        booz_jax.res = bad_res
        booz_jax.need_to_run_code = False
        try:
            obj = _build_boozer_wrapper(wrapper_name, booz_jax, bs_jax)
            assert np.isfinite(float(obj.J()))
            with pytest.raises(
                ValueError, match="requires fixed coil currents when G=None"
            ):
                obj.dJ()
        finally:
            booz_jax.res = old_res
            booz_jax.need_to_run_code = old_dirty


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

        logger.info(
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
        _assert_boozer_surfaces_end_state_parity(
            "CPU",
            exact_pair.booz_cpu_exact,
            "JAX CPU",
            exact_pair.booz_jax_exact,
            tolerances=_EXACT_SOLVER_END_STATE_TOLS,
            max_reference_residual_inf=_EXACT_SOLVER_RESIDUAL_INF_MAX,
            max_candidate_residual_inf=_EXACT_SOLVER_RESIDUAL_INF_MAX,
        )

    def test_value_wrappers_match_on_shared_exact_state(self):
        exact_pair = _solve_exact_cpu_jax_parity_pair()
        iota_abs_tol = 1e-12
        nqs_rel_tol = 1e-10
        nqs_abs_tol = 1e-12

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

    def test_gradient_wrappers_healthy_on_exact_state(self):
        """IotasJAX.dJ() and NonQuasiSymmetricRatioJAX.dJ() are healthy on exact surface.

        The exact Newton Jacobian is ill-conditioned: scipy and JAX PLU
        factorizations choose different pivots, producing adjoint vectors
        that both satisfy J^T adj = rhs to machine precision but differ in
        norm by ~3x.  This makes direct gradient parity impossible on the
        exact path.

        Direct gradient parity IS validated on the LS on-device path at
        rtol=1e-10 in test_real_fixture_ondevice_parity_and_wrapper_gradients.
        FD correctness is validated on the LS re-solve path (not exact) in
        TestIotasJAXResolveFD and TestNonQSRatioJAXResolveFD.  This test
        covers only gradient health (finite, non-zero) on the exact state.
        """
        exact_pair = _solve_exact_cpu_jax_parity_pair()

        iotas_cpu_grad = np.array(Iotas(exact_pair.booz_cpu_exact).dJ())
        iotas_jax_grad = np.array(IotasJAX(exact_pair.booz_jax_exact).dJ())
        assert iotas_cpu_grad.shape == iotas_jax_grad.shape
        _assert_gradients_finite_nonzero(
            [iotas_cpu_grad, iotas_jax_grad], "Exact-path Iotas.dJ()"
        )

        nqs_cpu_grad = np.array(
            NonQuasiSymmetricRatio(
                exact_pair.booz_cpu_exact, exact_pair.bs_cpu, sDIM=6
            ).dJ()
        )
        nqs_jax_grad = np.array(
            NonQuasiSymmetricRatioJAX(
                exact_pair.booz_jax_exact, exact_pair.bs_jax, sDIM=6
            ).dJ()
        )
        assert nqs_cpu_grad.shape == nqs_jax_grad.shape
        _assert_gradients_finite_nonzero(
            [nqs_cpu_grad, nqs_jax_grad], "Exact-path NonQuasiSymmetricRatio.dJ()"
        )

    def test_exact_coil_vjp_matches_fixed_state_directional_fd(self):
        exact_pair = _solve_exact_cpu_jax_parity_pair()
        booz_jax = exact_pair.booz_jax_exact
        bs_jax = exact_pair.bs_jax
        res_exact = exact_pair.res_jax
        iota = res_exact["iota"]
        G = res_exact["G"]
        fd_rel_tol = 1e-4
        fd_abs_tol = 1e-8

        lm = np.ones(res_exact["PLU"][1].shape[0], dtype=float)
        d_coil_arrays, coil_indices = res_exact["vjp"](
            lm,
            booz_jax,
            iota,
            G,
        )
        derivative = bs_jax.coil_cotangents_to_derivative(d_coil_arrays, coil_indices)
        full_gradient = np.asarray(derivative(bs_jax), dtype=float)

        directional_objective_at = _make_fixed_state_exact_directional_objective(
            booz_jax,
            bs_jax,
            lm,
            iota,
            G,
        )

        _assert_directional_derivative_matches_fd(
            full_gradient,
            directional_objective_at,
            np.asarray(bs_jax.x.copy(), dtype=float),
            rng_seed=3,
            eps=1e-5,
            num_directions=3,
            rel_tol=fd_rel_tol,
            abs_tol=fd_abs_tol,
            label="Exact cotangent FD",
        )

    def test_boozer_residual_wrapper_rejects_exact_surface(self):
        exact_pair = _solve_exact_cpu_jax_parity_pair()

        with pytest.raises(ValueError, match="least-squares BoozerSurfaceJAX"):
            BoozerResidualJAX(exact_pair.booz_jax_exact, exact_pair.bs_jax)


# -----------------------------------------------------------------------
# Test 16: IotasJAX re-solve FD on the stable reduced real fixture
# -----------------------------------------------------------------------


class TestIotasJAXResolveFD:
    """IotasJAX.dJ() matches central FD through the full re-solve path.

    Perturbs coil DOFs, re-runs the inner Boozer solve, and checks that the
    directional derivative predicted by IotasJAX.dJ() matches the finite-
    difference approximation of (iota(coils+eps) - iota(coils-eps)) / (2*eps).
    """

    @pytest.mark.slow
    def test_iotas_resolve_fd(self, real_resolve_fd_suite):
        _assert_wrapper_resolve_fd_matches_real_fixture(
            wrapper_label="IotasJAX",
            real_resolve_fd_suite=real_resolve_fd_suite,
        )


# -----------------------------------------------------------------------
# Test 17: NonQuasiSymmetricRatioJAX re-solve FD
# -----------------------------------------------------------------------


class TestNonQSRatioJAXResolveFD:
    """NonQuasiSymmetricRatioJAX.dJ() matches central FD through re-solve.

    Same pattern as TestIotasJAXResolveFD but for the QS ratio wrapper.
    """

    @pytest.mark.slow
    def test_nqsr_resolve_fd(self, real_resolve_fd_suite):
        _assert_wrapper_resolve_fd_matches_real_fixture(
            wrapper_label="NonQuasiSymmetricRatioJAX",
            real_resolve_fd_suite=real_resolve_fd_suite,
        )


# -----------------------------------------------------------------------
# Test 18: BoozerResidualJAX end-to-end adjoint FD (P27)
# -----------------------------------------------------------------------


class TestBoozerResidualAdjointFD:
    """BoozerResidualJAX.dJ() matches central FD through the full re-solve path.

    Unlike TestBoozerResidualGradientFD (which validates at fixed surface
    where adjoint ≈ 0), this test perturbs coil DOFs AND re-solves the
    inner Boozer system, so the FD naturally captures the adjoint
    contribution from surface movement.

    The gradient is the full IFT composed gradient:
        dJ/d_coils = dJ_direct - adj^T dg/d_coils

    This validates that the adjoint solve and VJP projection are correct
    end-to-end, not just at fixed surface.
    """

    @pytest.mark.slow
    def test_boozer_residual_resolve_fd(self, real_resolve_fd_suite):
        _assert_wrapper_resolve_fd_matches_real_fixture(
            wrapper_label="BoozerResidualJAX",
            real_resolve_fd_suite=real_resolve_fd_suite,
        )

    def test_adjoint_fraction_diagnostic(self):
        """Measure the adjoint fraction of the total BoozerResidualJAX gradient.

        This is a diagnostic that reports the adjoint fraction; it does NOT
        fail if the fraction is below 10%.  The re-solve FD above validates
        correctness regardless of the fraction.
        """
        from simsopt.geo.surfaceobjectives_jax import (
            _adjoint_coil_derivative,
            _current_coil_dofs_and_spec,
            _solve_boozer_adjoint,
            _value_and_direct_coil_derivative,
        )

        bs_jax, booz_jax, base_state = _make_real_resolve_fd_setup()
        jr = BoozerResidualJAX(booz_jax, bs_jax)
        jr.J()

        # Recompute with exposed intermediate terms
        booz_surf = jr.boozer_surface
        iota = booz_surf.res["iota"]
        G = booz_surf.res["G"]
        weight_inv_modB = booz_surf.res.get("weight_inv_modB", True)
        x_inner = booz_surf._pack_decision_vector(
            iota, G, sdofs=booz_surf._get_surface_dofs()
        )
        current_coil_dofs, coil_set_spec = _current_coil_dofs_and_spec(jr.biotsavart)
        _, direct_deriv = _value_and_direct_coil_derivative(
            jr.biotsavart,
            jr._direct_objective_value_and_grad,
            current_coil_dofs,
            x_inner,
            G is not None,
            weight_inv_modB,
        )
        dJ_ds = jr._compute_dJ_ds(coil_set_spec, iota, G, weight_inv_modB)
        adj = _solve_boozer_adjoint(booz_surf, dJ_ds)

        vjp_groups_fn = booz_surf.res.get("vjp_groups")
        adj_deriv = _adjoint_coil_derivative(
            vjp_groups_fn,
            booz_surf,
            iota,
            G,
            adj,
            jr.biotsavart,
        )

        direct_norm = np.linalg.norm(np.asarray(direct_deriv(jr.biotsavart)))
        adj_norm = np.linalg.norm(np.asarray(adj_deriv(jr.biotsavart)))
        total_norm = np.linalg.norm(np.asarray(jr.dJ()))
        adjoint_fraction = adj_norm / (direct_norm + 1e-30)

        logger.info(
            f"Adjoint fraction diagnostic:\n"
            f"  ||direct||  = {direct_norm:.6e}\n"
            f"  ||adjoint|| = {adj_norm:.6e}\n"
            f"  ||total||   = {total_norm:.6e}\n"
            f"  adjoint/direct = {adjoint_fraction:.4f}"
        )
        # Diagnostic only — the re-solve FD validates correctness regardless.
        # The fraction depends on how well-converged the inner solve is.
        assert total_norm > 0, "Total gradient is exactly zero"


# =======================================================================
# Section 3: Traceable Single-Stage Objective Path
# =======================================================================
#
# These tests define the contract for the traceable target-objective path
# inside the single-stage workflow so the outer optimizer can route through
# _minimize_lbfgs_private / _minimize_lbfgs_private_value_and_grad
# (lax.while_loop) on the supported path.
#
# Dependency order:
#   Test 3 (run_code_functional)
#     -> Test 1 (pure objective value)
#       -> Test 2 (jax.grad differentiable)
#         -> Test 4 (jaxpr traces without error)
#           -> Test 6 (routes through lax.while_loop)
#             -> Test 7 (parity with fused value/grad path)
#   Test 5 (no run_dict/Optimizable dependency) is independent
#
# This slice is green for the current traceable-objective path. Tests 1-7
# validate the pure array-backed custom_vjp objective built by
# make_traceable_objective(), while Tests 3a/3b continue to pin the lower-level
# legacy-result wrapper exposed by run_code_functional().
# This does not claim that the whole repo is fallback-free: reference/transitional
# optimizer lanes and compatibility adapters still exist elsewhere by design.
# =======================================================================


class TestRunCodeFunctional:
    """Test 3: BoozerSurfaceJAX.run_code_functional() — compatibility wrapper.

    The current run_code() mutates self state (need_to_run_code, surface DOFs
    via _set_surface_dofs), uses Python assertions, and branches on dirty flags.

    run_code_functional() must:
    - Accept explicit (coil_arrays, sdofs, iota, G) arguments
    - Return matching iota/G/success/PLU; s=None, vjp=None,
      vjp_groups=None (CPU callbacks incompatible with functional
      contract); sdofs=solved surface DOFs array
    - NOT mutate any self.* state

    Internally the solve now reuses run_code_traceable(), so the inner
    computation stays on the same pure-array target lane as the traceable
    single-stage objective. This wrapper still returns the historical
    run_code()-shaped dict, so the wrapper itself is not the JIT boundary.
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
            rtol=1e-9,
            atol=1e-12,
        )
        if res_stateful["G"] is not None:
            np.testing.assert_allclose(
                res_functional["G"],
                res_stateful["G"],
                rtol=1e-9,
                atol=1e-12,
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
    def _traceable_target_inputs(bs_jax, booz_jax, *, iota_target_shift=0.0):
        """Return the shared target-lane inputs used by traceable objective helpers."""
        iota_target = booz_jax.res["iota"] + iota_target_shift
        coil_dofs = jnp.array(bs_jax.x.copy())
        return iota_target, coil_dofs

    @staticmethod
    def _make_traceable(bs_jax, booz_jax, *, iota_target_shift=0.0):
        """Build the traceable objective and coil DOFs from a solved setup.

        Returns (f, coil_dofs, jr_jax, iotas_jax, iota_target).
        """
        jr_jax = BoozerResidualJAX(booz_jax, bs_jax)
        iotas_jax = IotasJAX(booz_jax)
        iota_target, coil_dofs = TestTraceableObjective._traceable_target_inputs(
            bs_jax,
            booz_jax,
            iota_target_shift=iota_target_shift,
        )

        from simsopt.geo.surfaceobjectives_jax import make_traceable_objective

        f = make_traceable_objective(booz_jax, bs_jax, iota_target)
        return f, coil_dofs, jr_jax, iotas_jax, iota_target

    @staticmethod
    def _make_traceable_value_and_grad(bs_jax, booz_jax, *, iota_target_shift=0.0):
        """Build the fused traceable value-and-gradient objective and coil DOFs."""
        jr_jax = BoozerResidualJAX(booz_jax, bs_jax)
        iotas_jax = IotasJAX(booz_jax)
        runtime_bundle, coil_dofs = (
            TestTraceableObjective._make_traceable_runtime_bundle(
                bs_jax,
                booz_jax,
                iota_target_shift=iota_target_shift,
            )
        )
        iota_target = booz_jax.res["iota"] + iota_target_shift
        fun_vg = runtime_bundle["value_and_grad"]
        return fun_vg, coil_dofs, jr_jax, iotas_jax, iota_target

    @staticmethod
    def _make_traceable_profile_suite(bs_jax, booz_jax, *, iota_target_shift=0.0):
        """Build the profiled traceable target-lane closure suite and coil DOFs."""
        runtime_bundle, coil_dofs = (
            TestTraceableObjective._make_traceable_runtime_bundle(
                bs_jax,
                booz_jax,
                iota_target_shift=iota_target_shift,
            )
        )
        profile_suite = runtime_bundle["profile_suite"]
        return profile_suite, coil_dofs

    @staticmethod
    def _make_traceable_runtime_bundle(
        bs_jax,
        booz_jax,
        *,
        iota_target_shift=0.0,
        include_profile_suite=True,
        success_filter=None,
    ):
        """Build the shared traceable runtime bundle and coil DOFs."""
        iota_target, coil_dofs = TestTraceableObjective._traceable_target_inputs(
            bs_jax,
            booz_jax,
            iota_target_shift=iota_target_shift,
        )

        from simsopt.geo.surfaceobjectives_jax import (
            make_traceable_objective_runtime_bundle,
        )

        runtime_bundle = make_traceable_objective_runtime_bundle(
            booz_jax,
            bs_jax,
            iota_target,
            include_profile_suite=include_profile_suite,
            success_filter=success_filter,
        )
        return runtime_bundle, coil_dofs

    @staticmethod
    def _assert_runtime_bundle_core_reused(runtime_bundle_a, runtime_bundle_b):
        assert runtime_bundle_a["objective"] is runtime_bundle_b["objective"]
        assert runtime_bundle_a["value_and_grad"] is runtime_bundle_b["value_and_grad"]

    @staticmethod
    def _assert_runtime_bundle_core_rebuilt(runtime_bundle_a, runtime_bundle_b):
        assert runtime_bundle_a["objective"] is not runtime_bundle_b["objective"]
        assert (
            runtime_bundle_a["value_and_grad"] is not runtime_bundle_b["value_and_grad"]
        )

    def test_runtime_bundle_success_filter_blocks_infeasible_nonbaseline_states(
        self,
        boozer_setup,
    ):
        """A target-lane success filter must demote infeasible candidate states."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        unconstrained_bundle, coil_dofs = self._make_traceable_runtime_bundle(
            bs_jax,
            booz_jax,
            include_profile_suite=False,
        )
        gated_bundle, _ = self._make_traceable_runtime_bundle(
            bs_jax,
            booz_jax,
            include_profile_suite=False,
            success_filter=lambda _coil_dofs, _solved_x: jnp.array(False, dtype=bool),
        )

        baseline_value = float(unconstrained_bundle["objective"](coil_dofs))
        gated_baseline_value = float(gated_bundle["objective"](coil_dofs))
        perturbed_coil_dofs = coil_dofs.at[0].add(
            jnp.asarray(1.0e-4, dtype=coil_dofs.dtype)
        )
        unconstrained_value = float(
            unconstrained_bundle["objective"](perturbed_coil_dofs)
        )
        gated_value = float(gated_bundle["objective"](perturbed_coil_dofs))
        gated_value_vg, gated_grad = gated_bundle["value_and_grad"](perturbed_coil_dofs)

        np.testing.assert_allclose(
            gated_baseline_value,
            baseline_value,
            rtol=0.0,
            atol=0.0,
            err_msg=(
                "The baseline target-lane state should remain evaluable even when "
                "the hard success filter rejects candidate states."
            ),
        )
        assert gated_value > unconstrained_value
        np.testing.assert_allclose(
            float(gated_value_vg),
            gated_value,
            rtol=0.0,
            atol=0.0,
            err_msg="Fused value_and_grad path must use the same gated failure value.",
        )
        assert np.all(np.isfinite(np.asarray(gated_grad)))

    def test_runtime_bundle_reuses_cached_compiled_callables(self, boozer_setup):
        """Repeated bundle construction should reuse the same compiled target-lane callables."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        runtime_bundle_a, _ = self._make_traceable_runtime_bundle(
            bs_jax,
            booz_jax,
            include_profile_suite=True,
        )
        runtime_bundle_b, _ = self._make_traceable_runtime_bundle(
            bs_jax,
            booz_jax,
            include_profile_suite=True,
        )

        self._assert_runtime_bundle_core_reused(runtime_bundle_a, runtime_bundle_b)
        assert (
            runtime_bundle_a["profile_suite"]["forward_value"]
            is runtime_bundle_b["profile_suite"]["forward_value"]
        )
        assert (
            runtime_bundle_a["profile_suite"]["field_eval"]
            is runtime_bundle_b["profile_suite"]["field_eval"]
        )
        assert (
            runtime_bundle_a["profile_suite"]["field_eval_sharding"]
            is runtime_bundle_b["profile_suite"]["field_eval_sharding"]
        )

    def test_runtime_bundle_rebuilds_when_target_changes(self, boozer_setup):
        """Changing the target objective inputs must invalidate the cached runtime bundle."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        runtime_bundle_a, _ = self._make_traceable_runtime_bundle(
            bs_jax,
            booz_jax,
            include_profile_suite=False,
        )
        runtime_bundle_b, _ = self._make_traceable_runtime_bundle(
            bs_jax,
            booz_jax,
            iota_target_shift=0.05,
            include_profile_suite=False,
        )

        self._assert_runtime_bundle_core_rebuilt(runtime_bundle_a, runtime_bundle_b)

    def test_runtime_bundle_rebuilds_after_solver_option_change_post_compile(
        self,
        boozer_setup,
    ):
        """Changing the traced inner-solver options must invalidate the cached bundle."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        runtime_bundle_a, coil_dofs = self._make_traceable_runtime_bundle(
            bs_jax,
            booz_jax,
            include_profile_suite=False,
        )

        first_value = runtime_bundle_a["objective"](coil_dofs)
        assert np.isfinite(float(first_value))

        original_method = booz_jax._resolve_optimizer_method()
        original_algorithm = booz_jax.options["least_squares_algorithm"]
        booz_jax.options["least_squares_algorithm"] = (
            "quasi-newton" if original_algorithm == "lm" else "lm"
        )
        booz_jax.options["limited_memory"] = False

        assert booz_jax._resolve_optimizer_method() != original_method

        runtime_bundle_b, _ = self._make_traceable_runtime_bundle(
            bs_jax,
            booz_jax,
            include_profile_suite=False,
        )

        self._assert_runtime_bundle_core_rebuilt(runtime_bundle_a, runtime_bundle_b)
        second_value = runtime_bundle_b["objective"](coil_dofs)
        second_value_vg, second_grad = runtime_bundle_b["value_and_grad"](coil_dofs)

        assert np.isfinite(float(second_value))
        np.testing.assert_allclose(
            float(second_value_vg),
            float(second_value),
            rtol=0.0,
            atol=0.0,
        )
        assert np.all(np.isfinite(np.asarray(second_grad)))

    def test_single_stage_hardware_success_filter_uses_cached_pytree_extraction_state(
        self,
        monkeypatch,
    ):
        """The target-lane feasibility filter should not reenter BiotSavart extraction methods."""
        (
            _coils,
            _surf_cpu,
            _surf_jax,
            _bs_cpu,
            bs_jax,
            _booz_cpu,
            booz_jax,
            _vol_cpu,
            iota0,
            G0,
        ) = _make_boozer_setup(constraint_weight=1.0)

        plasma_surface = SurfaceRZFourier(
            nfp=2,
            stellsym=True,
            mpol=1,
            ntor=0,
            quadpoints_phi=np.linspace(0.0, 0.5, 5, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False),
        )
        plasma_surface.set_rc(0, 0, 1.0)
        plasma_surface.set_rc(1, 0, 0.15)
        plasma_surface.set_zs(1, 0, 0.15)

        vessel_surface = SurfaceRZFourier(
            nfp=plasma_surface.nfp,
            stellsym=plasma_surface.stellsym,
            mpol=1,
            ntor=0,
            quadpoints_phi=plasma_surface.quadpoints_phi,
            quadpoints_theta=plasma_surface.quadpoints_theta,
        )
        vessel_surface.set_rc(0, 0, 2.0)
        vessel_surface.set_rc(1, 0, 0.5)
        vessel_surface.set_zs(1, 0, 0.5)
        booz_jax.res = {"G": jnp.asarray(G0, dtype=jnp.float64)}

        success_filter = (
            single_stage_example.build_single_stage_target_lane_hardware_success_filter(
                booz_jax,
                bs_jax,
                bs_jax.coils[0].curve,
                vessel_surface,
                cc_dist=0.0,
                cs_dist=0.0,
                ss_dist=0.0,
                curvature_threshold=1.0e9,
            )
        )

        solved_x = jnp.concatenate(
            (
                jnp.asarray(booz_jax.surface.get_dofs(), dtype=jnp.float64),
                jnp.asarray([iota0, G0], dtype=jnp.float64),
            )
        )

        def _reject_reconstruction(*_args, **_kwargs):
            raise AssertionError(
                "Single-stage hardware success filter should use cached immutable "
                "pytree extraction state after construction"
            )

        monkeypatch.setattr(bs_jax, "coil_set_spec_from_dofs", _reject_reconstruction)
        monkeypatch.setattr(bs_jax, "coil_specs_from_dofs", _reject_reconstruction)

        feasible = success_filter(
            jnp.asarray(bs_jax.x.copy(), dtype=jnp.float64),
            solved_x,
        )

        assert bool(np.asarray(feasible))

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
            atol=_TRACEABLE_OBJECTIVE_ABS_TOL,
            err_msg="Traceable objective value differs from JF.J()",
        )

    def test_pure_objective_matches_optimizable_value_with_offset_iota_target(
        self,
        boozer_setup,
    ):
        """Test 1b: nonzero iota penalty is included in the pure forward path."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        f, coil_dofs, jr_jax, iotas_jax, iota_target = self._make_traceable(
            bs_jax,
            booz_jax,
            iota_target_shift=1.0e-3,
        )

        JF_jax = jr_jax + QuadraticPenalty(iotas_jax, iota_target)
        j_reference = JF_jax.J()

        np.testing.assert_allclose(
            float(f(coil_dofs)),
            j_reference,
            rtol=1e-10,
            atol=_TRACEABLE_OBJECTIVE_ABS_TOL,
            err_msg="Traceable objective dropped the offset iota penalty",
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

    def test_traceable_value_and_grad_builder_matches_scalar_builder(
        self, boozer_setup
    ):
        """The fused target-lane builder must match the scalar traceable contract."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        f, coil_dofs, _, _, _ = self._make_traceable(bs_jax, booz_jax)
        fun_vg, _, _, _, _ = self._make_traceable_value_and_grad(bs_jax, booz_jax)

        value_scalar = f(coil_dofs)
        grad_scalar = jax.grad(f)(coil_dofs)
        value_vg, grad_vg = fun_vg(coil_dofs)

        np.testing.assert_allclose(
            np.asarray(value_vg),
            np.asarray(value_scalar),
            rtol=1e-10,
            atol=_TRACEABLE_OBJECTIVE_ABS_TOL,
        )
        np.testing.assert_allclose(
            np.asarray(grad_vg),
            np.asarray(grad_scalar),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_traceable_profile_suite_warmstart_predict_executes(self, boozer_setup):
        """The profiling suite must execute warmstart prediction without NameError."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        profile_suite, coil_dofs = self._make_traceable_profile_suite(bs_jax, booz_jax)

        warmstart_x = profile_suite["warmstart_predict"](coil_dofs)

        assert warmstart_x.shape[0] > 0
        assert jnp.all(jnp.isfinite(warmstart_x))

    def test_traceable_runtime_bundle_profiles_exact_optimizer_callable(
        self,
        boozer_setup,
    ):
        """The profile suite should expose the exact fused callable used by the optimizer."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        runtime_bundle, _ = self._make_traceable_runtime_bundle(bs_jax, booz_jax)

        assert (
            runtime_bundle["profile_suite"]["value_and_grad_pipeline"]
            is runtime_bundle["value_and_grad"]
        )

    def test_traceable_runtime_bundle_exposes_host_normalized_wrappers(
        self,
        boozer_setup,
    ):
        """The runtime bundle should expose explicit host-boundary companions."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        runtime_bundle, coil_dofs = self._make_traceable_runtime_bundle(
            bs_jax,
            booz_jax,
            include_profile_suite=False,
        )

        pure_value = runtime_bundle["objective"](coil_dofs)
        pure_value_vg, pure_grad = runtime_bundle["value_and_grad"](coil_dofs)
        host_value = runtime_bundle["host_objective"](coil_dofs)
        host_value_vg, host_grad = runtime_bundle["host_value_and_grad"](coil_dofs)

        assert isinstance(pure_value, jax.Array)
        assert np.asarray(pure_value).shape == ()
        assert isinstance(pure_value_vg, jax.Array)
        assert np.asarray(pure_value_vg).shape == ()
        assert isinstance(pure_grad, jax.Array)
        assert isinstance(host_value, float)
        assert isinstance(host_value_vg, float)
        assert isinstance(host_grad, np.ndarray)

        np.testing.assert_allclose(host_value, float(pure_value), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            host_value_vg,
            float(pure_value_vg),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            host_grad,
            np.asarray(pure_grad),
            rtol=0.0,
            atol=0.0,
        )

    def test_traceable_profile_suite_field_eval_sharding_reuses_compiled_pipeline(
        self,
        boozer_setup,
        monkeypatch,
    ):
        import simsopt.geo.surfaceobjectives_jax as soj

        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        runtime_bundle, coil_dofs = self._make_traceable_runtime_bundle(
            bs_jax,
            booz_jax,
            include_profile_suite=True,
        )
        field_eval_sharding = runtime_bundle["profile_suite"]["field_eval_sharding"]
        original_jit = soj.jax.jit
        jit_call_count = 0

        def counting_jit(*args, **kwargs):
            nonlocal jit_call_count
            jit_call_count += 1
            return original_jit(*args, **kwargs)

        monkeypatch.setattr(soj.jax, "jit", counting_jit)

        summary_a = field_eval_sharding(coil_dofs)
        first_call_jit_count = jit_call_count
        summary_b = field_eval_sharding(coil_dofs)

        assert summary_a == summary_b
        assert jit_call_count == first_call_jit_count

    def test_traceable_runtime_bundle_skips_profile_suite_by_default(
        self, boozer_setup
    ):
        """The default runtime bundle should avoid building profiling siblings."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        runtime_bundle, _ = self._make_traceable_runtime_bundle(
            bs_jax,
            booz_jax,
            include_profile_suite=False,
        )

        assert "value_and_grad" in runtime_bundle
        assert "profile_suite" not in runtime_bundle

    def test_pure_objective_traces_to_jaxpr(self, boozer_setup):
        """Test 4: jax.make_jaxpr succeeds without a callback bridge."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        f, coil_dofs, _, _, _ = self._make_traceable(bs_jax, booz_jax)

        jaxpr = jax.make_jaxpr(f)(coil_dofs)
        assert jaxpr is not None, "make_jaxpr returned None"
        assert "pure_callback" not in str(jaxpr), (
            "Traceable objective still routes through jax.pure_callback"
        )

    def test_traceable_runtime_bundle_matches_sharded_field_contract(
        self,
        boozer_setup,
        monkeypatch,
    ):
        import simsopt.jax_core.sharding as sharding_core

        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        runtime_bundle, coil_dofs = self._make_traceable_runtime_bundle(
            bs_jax,
            booz_jax,
            include_profile_suite=True,
        )

        dense_value = runtime_bundle["objective"](coil_dofs)
        dense_value_vg, dense_grad = runtime_bundle["value_and_grad"](coil_dofs)

        monkeypatch.setattr(
            sharding_core,
            "get_sharding_tuning",
            lambda mode=None: types.SimpleNamespace(
                active=True,
                strategy="hybrid",
                min_points_to_shard=1,
                min_pairwise_rows_to_shard=1,
                platform="cpu",
                mesh_axis_name="d",
            ),
        )

        sharded_value = runtime_bundle["objective"](coil_dofs)
        sharded_value_vg, sharded_grad = runtime_bundle["value_and_grad"](coil_dofs)
        summary = runtime_bundle["profile_suite"]["field_eval_sharding"](coil_dofs)

        np.testing.assert_allclose(
            float(sharded_value),
            float(dense_value),
            rtol=1e-10,
            atol=_TRACEABLE_OBJECTIVE_ABS_TOL,
        )
        np.testing.assert_allclose(
            float(sharded_value_vg),
            float(dense_value_vg),
            rtol=1e-10,
            atol=_TRACEABLE_OBJECTIVE_ABS_TOL,
        )
        np.testing.assert_allclose(
            np.asarray(sharded_grad),
            np.asarray(dense_grad),
            rtol=1e-10,
            atol=1e-10,
        )
        assert summary["kind"] in {"NamedSharding", "SingleDeviceSharding"}
        assert summary["device_count"] >= 1

    def test_traceable_objective_accepts_lm_ondevice_inner_solve(self, boozer_setup):
        """The single-stage target lane must allow the LM Boozer inner solve."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        booz_jax.options["least_squares_algorithm"] = "lm"
        runtime_bundle, coil_dofs = self._make_traceable_runtime_bundle(
            bs_jax,
            booz_jax,
            include_profile_suite=False,
        )

        value = runtime_bundle["objective"](coil_dofs)
        value_vg, grad = runtime_bundle["value_and_grad"](coil_dofs)
        jaxpr = jax.make_jaxpr(runtime_bundle["objective"])(coil_dofs)

        assert np.isfinite(float(value))
        np.testing.assert_allclose(float(value_vg), float(value), rtol=0.0, atol=0.0)
        assert np.all(np.isfinite(np.asarray(grad)))
        assert jaxpr is not None

    def test_traceable_objective_does_not_reenter_host_snapshot_after_build(
        self,
        boozer_setup,
        monkeypatch,
    ):
        """Target-lane evaluation must stay off the host after bundle construction."""
        import simsopt.geo.boozersurface_jax as bsj

        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        runtime_bundle, coil_dofs = self._make_traceable_runtime_bundle(
            bs_jax,
            booz_jax,
            include_profile_suite=False,
        )

        def _reject_host_snapshot(*_args, **_kwargs):
            raise AssertionError(
                "Single-stage ondevice objective should not hostify mutable state "
                "inside the compiled hot path."
            )

        monkeypatch.setattr(bsj, "_hostify_tree", _reject_host_snapshot)

        value = runtime_bundle["objective"](coil_dofs)
        value_vg, grad = runtime_bundle["value_and_grad"](coil_dofs)
        jaxpr = jax.make_jaxpr(runtime_bundle["objective"])(coil_dofs)

        assert np.isfinite(float(value))
        np.testing.assert_allclose(
            float(value_vg),
            float(value),
            rtol=0.0,
            atol=0.0,
        )
        assert np.all(np.isfinite(np.asarray(grad)))
        assert jaxpr is not None

    def test_traceable_runtime_bundle_does_not_call_biotsavart_reconstruction_methods_after_build(
        self,
        boozer_setup,
        monkeypatch,
    ):
        """The compiled target lane should use cached pytree extraction state only."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        runtime_bundle, coil_dofs = self._make_traceable_runtime_bundle(
            bs_jax,
            booz_jax,
            include_profile_suite=False,
        )

        def _reject_reconstruction(*_args, **_kwargs):
            raise AssertionError(
                "Single-stage runtime bundle should not call BiotSavartJAX "
                "reconstruction helpers after bundle construction"
            )

        monkeypatch.setattr(bs_jax, "coil_set_spec_from_dofs", _reject_reconstruction)
        monkeypatch.setattr(bs_jax, "coil_specs_from_dofs", _reject_reconstruction)

        value = runtime_bundle["objective"](coil_dofs)
        value_vg, grad = runtime_bundle["value_and_grad"](coil_dofs)

        assert np.isfinite(float(value))
        np.testing.assert_allclose(float(value_vg), float(value), rtol=0.0, atol=0.0)
        assert np.all(np.isfinite(np.asarray(grad)))

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

    def test_traceable_scalar_routes_through_lax_while_loop(
        self,
        boozer_setup,
        monkeypatch,
    ):
        """Test 6: scalar lbfgs-ondevice stays on the private ondevice path."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        f, x0, _, _, _ = self._make_traceable(bs_jax, booz_jax)

        import simsopt.geo.optimizer_jax as opt_mod

        assert not hasattr(opt_mod, "_minimize_lbfgs_explicit_value_and_grad")
        original = opt_mod._minimize_lbfgs_private
        calls = {"count": 0}

        def _counting_private(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(opt_mod, "_minimize_lbfgs_private", _counting_private)

        result = jax_minimize(
            f,
            x0,
            method="lbfgs-ondevice",
            maxiter=2,
            tol=1e-20,
        )
        assert calls["count"] == 1
        assert np.isfinite(float(result.fun)), "Optimizer produced non-finite J"

    def test_traceable_value_and_grad_routes_through_ondevice_private_path(
        self,
        boozer_setup,
        monkeypatch,
    ):
        """Test 6b: explicit value/grad lbfgs-ondevice uses the private fused path."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        fun_vg, x0, _, _, _ = self._make_traceable_value_and_grad(bs_jax, booz_jax)

        import simsopt.geo.optimizer_jax as opt_mod

        assert not hasattr(opt_mod, "_minimize_lbfgs_explicit_value_and_grad")
        original = opt_mod._minimize_lbfgs_private_value_and_grad
        calls = {"count": 0}

        def _counting_private(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            opt_mod,
            "_minimize_lbfgs_private_value_and_grad",
            _counting_private,
        )

        result = jax_minimize(
            fun_vg,
            x0,
            method="lbfgs-ondevice",
            value_and_grad=True,
            maxiter=2,
            tol=1e-20,
        )
        assert calls["count"] == 1
        assert np.isfinite(float(result.fun)), "Optimizer produced non-finite J"

    def test_traceable_matches_fused_value_and_grad_path(self, boozer_setup):
        """Test 7: Traceable scalar and fused value/grad paths produce same J."""
        (_, _, _, _, bs_jax, _, booz_jax, _) = boozer_setup
        f, x0, _, _, _ = self._make_traceable(bs_jax, booz_jax)
        fun_vg, _, _, _, _ = self._make_traceable_value_and_grad(bs_jax, booz_jax)

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

            result_b = jax_minimize(
                fun_vg,
                x0,
                method="lbfgs-ondevice",
                value_and_grad=True,
                maxiter=3,
                tol=1e-20,
            )

            # The compared objective values are nominally zero at this point in
            # the short run, so a tiny absolute floor is more meaningful than a
            # pure relative check.
            np.testing.assert_allclose(
                float(result_a.fun),
                float(result_b.fun),
                rtol=1e-10,
                atol=_TRACEABLE_OBJECTIVE_ABS_TOL,
                err_msg=(
                    f"Traceable J={float(result_a.fun):.6e} vs "
                    f"explicit J={float(result_b.fun):.6e}"
                ),
            )
        finally:
            _restore_state()


# -----------------------------------------------------------------------
# Test P15: Boozer residual direct CPU parity
# -----------------------------------------------------------------------


class TestBoozerResidualCPUParity:
    """Direct comparison of JAX boozer_residual_scalar against the C++
    sopp.boozer_residual at the same (surface, iota, G) state.

    This test requires simsoptpp for the CPU reference.  Previously the JAX
    Boozer residual was only validated via FD convergence; this class
    establishes direct numerical parity with the C++ kernel.
    """

    @staticmethod
    def _build_shared_state(seed=42):
        """Build a shared (surface, coils, iota, G) state for parity evaluation.

        Returns arrays that both CPU and JAX kernels can consume, ensuring
        the comparison is at exactly the same numerical state.
        """
        np.random.seed(seed)

        ncoils, nfp = 2, 2
        stellsym = True
        base_curves = create_equally_spaced_curves(
            ncoils, nfp, stellsym=stellsym, R0=1.0, R1=0.5, order=3
        )
        base_currents = [Current(1e5) for _ in range(ncoils)]
        for c in base_currents:
            c.fix_all()
        coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)

        mpol, ntor = 2, 2
        nphi = 2 * ntor + 1
        ntheta = 2 * mpol + 1
        surf = SurfaceXYZTensorFourier(
            mpol=mpol,
            ntor=ntor,
            stellsym=stellsym,
            nfp=nfp,
            quadpoints_phi=np.linspace(0, 1.0 / nfp, nphi, endpoint=False),
            quadpoints_theta=np.linspace(0, 1.0, ntheta, endpoint=False),
        )
        surf.set_dofs(np.zeros_like(surf.get_dofs()))

        # Fit to a simple RZ surface so the geometry is non-trivial
        s_rz = SurfaceRZFourier(
            nfp=nfp,
            stellsym=stellsym,
            mpol=1,
            ntor=0,
            quadpoints_phi=surf.quadpoints_phi,
            quadpoints_theta=surf.quadpoints_theta,
        )
        s_rz.set_rc(0, 0, 1.0)
        s_rz.set_rc(1, 0, 0.15)
        s_rz.set_zs(1, 0, 0.15)
        surf.least_squares_fit(s_rz.gamma())

        # Add a small random perturbation so the residual is non-trivial
        dofs = surf.get_dofs()
        dofs += np.random.randn(len(dofs)) * 1e-4
        surf.set_dofs(dofs)

        mu0 = 4 * np.pi * 1e-7
        G = mu0 * sum(abs(c.current.get_value()) for c in coils)
        iota = 0.3

        # Evaluate surface geometry and B-field via CPU objects
        gamma = surf.gamma()
        xphi = surf.gammadash1()
        xtheta = surf.gammadash2()

        bs_cpu = BiotSavart(coils)
        xsemiflat = gamma.reshape(-1, 3)
        bs_cpu.set_points(xsemiflat)
        B = bs_cpu.B().reshape(gamma.shape)

        return gamma, xphi, xtheta, B, iota, G, surf, coils, bs_cpu

    def test_raw_boozer_residual_scalar_parity(self):
        """JAX boozer_residual_scalar matches sopp.boozer_residual / num_res.

        Compares the core Boozer residual scalar (without label or z
        constraints) at a shared surface state with a small random
        perturbation so the residual value is non-trivial.
        """
        from simsopt.geo.boozer_residual_jax import boozer_residual_scalar

        gamma, xphi, xtheta, B, iota, G, surf, coils, bs_cpu = self._build_shared_state(
            seed=42
        )
        nphi, ntheta, _ = B.shape
        num_res = 3 * nphi * ntheta

        # CPU: sopp.boozer_residual returns the raw half-sum-of-squares
        val_cpu = sopp.boozer_residual(G, iota, xphi, xtheta, B, True)
        val_cpu_normalized = val_cpu / num_res

        # JAX: boozer_residual_scalar includes the / num_res normalization
        val_jax = float(
            boozer_residual_scalar(
                G,
                iota,
                jnp.array(B),
                jnp.array(xphi),
                jnp.array(xtheta),
                weight_inv_modB=True,
            )
        )

        logger.info(
            f"Boozer residual scalar parity (weight_inv_modB=True):\n"
            f"  CPU: {val_cpu_normalized:.15e}\n"
            f"  JAX: {val_jax:.15e}\n"
            f"  |diff|: {abs(val_cpu_normalized - val_jax):.3e}"
        )

        assert val_cpu_normalized > 1e-10, (
            f"CPU residual unexpectedly tiny ({val_cpu_normalized:.3e}); "
            "perturbation may not have taken effect"
        )
        np.testing.assert_allclose(
            val_jax,
            val_cpu_normalized,
            rtol=1e-10,
            err_msg="JAX boozer_residual_scalar does not match C++ sopp.boozer_residual",
        )

    def test_raw_boozer_residual_scalar_parity_no_weight(self):
        """Same as above but with weight_inv_modB=False."""
        from simsopt.geo.boozer_residual_jax import boozer_residual_scalar

        gamma, xphi, xtheta, B, iota, G, surf, coils, bs_cpu = self._build_shared_state(
            seed=43
        )
        nphi, ntheta, _ = B.shape
        num_res = 3 * nphi * ntheta

        val_cpu = sopp.boozer_residual(G, iota, xphi, xtheta, B, False)
        val_cpu_normalized = val_cpu / num_res

        val_jax = float(
            boozer_residual_scalar(
                G,
                iota,
                jnp.array(B),
                jnp.array(xphi),
                jnp.array(xtheta),
                weight_inv_modB=False,
            )
        )

        logger.info(
            f"Boozer residual scalar parity (weight_inv_modB=False):\n"
            f"  CPU: {val_cpu_normalized:.15e}\n"
            f"  JAX: {val_jax:.15e}\n"
            f"  |diff|: {abs(val_cpu_normalized - val_jax):.3e}"
        )

        assert val_cpu_normalized > 1e-10, (
            f"CPU residual unexpectedly tiny ({val_cpu_normalized:.3e})"
        )
        np.testing.assert_allclose(
            val_jax,
            val_cpu_normalized,
            rtol=1e-10,
            err_msg=(
                "JAX boozer_residual_scalar (no weight) does not match "
                "C++ sopp.boozer_residual"
            ),
        )

    def test_boozer_residual_vector_parity(self):
        """JAX boozer_residual_vector matches the component-wise C++ residual.

        The C++ sopp.boozer_residual_ds with derivatives=0 only returns the
        scalar, so we verify vector parity indirectly: the JAX vector's
        squared norm (0.5 ||r||^2 / num_res) must equal the scalar from both
        sides.
        """
        from simsopt.geo.boozer_residual_jax import (
            boozer_residual_scalar,
            boozer_residual_vector,
        )

        gamma, xphi, xtheta, B, iota, G, surf, coils, bs_cpu = self._build_shared_state(
            seed=44
        )
        nphi, ntheta, _ = B.shape
        num_res = 3 * nphi * ntheta

        B_jax = jnp.array(B)
        xphi_jax = jnp.array(xphi)
        xtheta_jax = jnp.array(xtheta)

        r_vec = boozer_residual_vector(
            G, iota, B_jax, xphi_jax, xtheta_jax, weight_inv_modB=True
        )
        scalar_from_vec = float(0.5 * jnp.sum(r_vec**2) / num_res)

        scalar_direct = float(
            boozer_residual_scalar(
                G, iota, B_jax, xphi_jax, xtheta_jax, weight_inv_modB=True
            )
        )

        val_cpu = sopp.boozer_residual(G, iota, xphi, xtheta, B, True) / num_res

        logger.info(
            f"Vector consistency:\n"
            f"  0.5||r||^2/N from vector: {scalar_from_vec:.15e}\n"
            f"  boozer_residual_scalar:    {scalar_direct:.15e}\n"
            f"  sopp.boozer_residual/N:    {val_cpu:.15e}"
        )

        np.testing.assert_allclose(
            scalar_from_vec,
            scalar_direct,
            rtol=1e-12,
            err_msg="JAX vector norm disagrees with JAX scalar",
        )
        np.testing.assert_allclose(
            scalar_from_vec,
            val_cpu,
            rtol=1e-10,
            err_msg="JAX vector norm disagrees with C++ scalar",
        )

    def test_full_penalty_objective_parity(self):
        """Full penalty objective (Boozer + label + z) matches CPU vs JAX.

        Uses the same (surface, iota, G) state and compares the CPU
        boozer_penalty_constraints_vectorized output against the JAX
        _boozer_penalty_objective at the initial guess (no solving).
        """
        _gamma, _xphi, _xtheta, _B, iota0, _G, surf_cpu, coils_list, bs_cpu = (
            self._build_shared_state(seed=45)
        )

        # Duplicate surface for JAX side
        surf_jax = SurfaceXYZTensorFourier(
            mpol=surf_cpu.mpol,
            ntor=surf_cpu.ntor,
            stellsym=surf_cpu.stellsym,
            nfp=surf_cpu.nfp,
            quadpoints_phi=surf_cpu.quadpoints_phi,
            quadpoints_theta=surf_cpu.quadpoints_theta,
        )
        surf_jax.set_dofs(surf_cpu.get_dofs().copy())

        bs_jax = BiotSavartJAX(coils_list)

        vol_cpu = Volume(surf_cpu)
        vol_jax = Volume(surf_jax)
        vol_target = vol_cpu.J()

        constraint_weight = 1.0

        # CPU: evaluate full penalty via BoozerSurface
        booz_cpu = BoozerSurface(
            bs_cpu,
            surf_cpu,
            vol_cpu,
            vol_target,
            constraint_weight=constraint_weight,
            options={"verbose": False},
        )
        x_cpu = np.concatenate((surf_cpu.get_dofs(), [iota0]))
        val_cpu = booz_cpu.boozer_penalty_constraints_vectorized(
            x_cpu,
            derivatives=0,
            constraint_weight=constraint_weight,
            optimize_G=False,
            weight_inv_modB=True,
        )

        # JAX: evaluate full penalty via _boozer_penalty_objective
        booz_jax = BoozerSurfaceJAX(
            bs_jax,
            surf_jax,
            vol_jax,
            vol_target,
            constraint_weight=constraint_weight,
            options={
                "verbose": False,
                "weight_inv_modB": True,
            },
        )
        coil_arrays = booz_jax._coil_arrays
        obj_fn = _make_ls_penalty_objective(
            booz_jax,
            coil_arrays,
            optimize_G=False,
            weight_inv_modB=True,
        )
        x_jax = jnp.concatenate(
            [
                jnp.array(surf_jax.get_dofs()),
                jnp.array([iota0]),
            ]
        )
        val_jax = float(obj_fn(x_jax))

        logger.info(
            f"Full penalty objective parity:\n"
            f"  CPU: {val_cpu:.15e}\n"
            f"  JAX: {val_jax:.15e}\n"
            f"  |diff|: {abs(val_cpu - val_jax):.3e}"
        )

        np.testing.assert_allclose(
            val_jax,
            val_cpu,
            rtol=1e-10,
            err_msg=(
                "Full JAX penalty objective does not match CPU "
                "boozer_penalty_constraints_vectorized"
            ),
        )
