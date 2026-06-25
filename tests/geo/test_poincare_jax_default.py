import types

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from examples.single_stage_optimization.POINCARE_PLOTTING import (
    poincare_surfaces_jax_default as module,
)
from examples.single_stage_optimization.topology_scorer import trace_metrics


def test_biot_savart_jax_matches_circular_loop_axis_field():
    radius = 1.7
    current = 2.3
    z = 0.4
    nquad = 4096
    t = np.linspace(0.0, 1.0, nquad, endpoint=False)
    angle = 2.0 * np.pi * t
    gamma = np.stack(
        (
            radius * np.cos(angle),
            radius * np.sin(angle),
            np.zeros_like(angle),
        ),
        axis=1,
    )
    gammadash = np.stack(
        (
            -2.0 * np.pi * radius * np.sin(angle),
            2.0 * np.pi * radius * np.cos(angle),
            np.zeros_like(angle),
        ),
        axis=1,
    )
    field = module._biot_savart_points_jax(
        jnp.asarray([[0.0, 0.0, z]], dtype=jnp.float64),
        jnp.asarray(gamma[None, :, :], dtype=jnp.float64),
        jnp.asarray(gammadash[None, :, :], dtype=jnp.float64),
        jnp.ones((1, nquad), dtype=jnp.float64),
        jnp.asarray([current], dtype=jnp.float64),
    )
    expected_bz = (
        4.0e-7
        * np.pi
        * current
        * radius**2
        / (2.0 * (radius**2 + z**2) ** 1.5)
    )
    np.testing.assert_allclose(
        np.asarray(field)[0],
        [0.0, 0.0, expected_bz],
        rtol=1e-5,
        atol=1e-12,
    )


def test_biot_savart_jax_ignores_masked_padding_samples():
    radius = 1.2
    current = 1.4
    z = 0.3
    nquad = 128
    padded_nquad = 256
    t = np.linspace(0.0, 1.0, nquad, endpoint=False)
    angle = 2.0 * np.pi * t
    gamma = np.stack(
        (
            radius * np.cos(angle),
            radius * np.sin(angle),
            np.zeros_like(angle),
        ),
        axis=1,
    )
    gammadash = np.stack(
        (
            -2.0 * np.pi * radius * np.sin(angle),
            2.0 * np.pi * radius * np.cos(angle),
            np.zeros_like(angle),
        ),
        axis=1,
    )
    padded_gamma = np.full((1, padded_nquad, 3), 99.0)
    padded_gammadash = np.full((1, padded_nquad, 3), -77.0)
    padded_gamma[0, :nquad, :] = gamma
    padded_gammadash[0, :nquad, :] = gammadash
    mask = np.zeros((1, padded_nquad), dtype=float)
    mask[0, :nquad] = 1.0

    padded = module._biot_savart_points_jax(
        jnp.asarray([[0.0, 0.0, z]], dtype=jnp.float64),
        jnp.asarray(padded_gamma, dtype=jnp.float64),
        jnp.asarray(padded_gammadash, dtype=jnp.float64),
        jnp.asarray(mask, dtype=jnp.float64),
        jnp.asarray([current], dtype=jnp.float64),
    )
    expected_bz = (
        4.0e-7
        * np.pi
        * current
        * radius**2
        / (2.0 * (radius**2 + z**2) ** 1.5)
    )

    np.testing.assert_allclose(
        np.asarray(padded)[0],
        [0.0, 0.0, expected_bz],
        rtol=1e-5,
        atol=1e-12,
    )


def test_jax_trace_settings_require_four_plane_alignment():
    assert module.validate_jax_trace_settings(3, 64) == 16
    with pytest.raises(ValueError, match="divisible by 4"):
        module.validate_jax_trace_settings(3, 62)


def test_jax_trace_records_match_topology_metrics_contract():
    trace_result = {
        "hit_xyz": np.asarray(
            [
                [[1.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 1.1, 0.0]],
                [[-1.0, 0.0, 0.0], [-1.1, 0.0, 0.0]],
                [[0.0, -1.0, 0.0], [0.0, -1.1, 0.0]],
            ],
            dtype=float,
        ),
        "hit_alive": np.asarray(
            [
                [True, True],
                [True, True],
                [True, False],
                [True, False],
            ],
            dtype=bool,
        ),
        "hit_time": np.asarray([0.0, 0.25, 0.5, 0.75], dtype=float),
        "hit_phi_index": np.asarray([0, 1, 2, 3], dtype=int),
        "stop_time": np.asarray([np.nan, 0.4], dtype=float),
        "stop_xyz": np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 1.2, 0.0]],
            dtype=float,
        ),
        "stop_reason": np.asarray([-1, 0], dtype=int),
        "final_states": np.asarray([[1.0, 0.0], [1.2, 0.0]], dtype=float),
        "final_alive": np.asarray([True, False], dtype=bool),
        "final_phi": np.asarray(1.0, dtype=float),
    }
    fieldlines_tys, fieldlines_phi_hits, phis = (
        module.jax_trace_records_to_simsopt_format(trace_result, nfp=2)
    )
    metrics = trace_metrics(
        fieldlines_tys,
        fieldlines_phi_hits,
        phis,
        module.JAX_DEFAULT_STOP_LABELS,
        mode="default",
    )

    assert metrics["validation_status"] == "diagnostic_only"
    assert metrics["survived_lines"] == 1
    assert metrics["stop_reason_counts"]["max_z_guardrail"] == 1
    assert metrics["line_metrics"][1]["first_exit_time"] == pytest.approx(0.4)


def test_jax_invalid_field_stop_is_not_reported_as_iteration_limit():
    states = jnp.asarray([[1.0, 0.0]], dtype=jnp.float64)
    in_bounds, stop_reason = module._box_guard_status(
        states,
        jnp.asarray([False]),
        rmin=0.5,
        rmax=1.5,
        zmax=0.5,
    )

    assert not bool(np.asarray(in_bounds)[0])
    assert int(np.asarray(stop_reason)[0]) == module.PHI_ODE_SINGULARITY_STOP_REASON

    trace_result = {
        "hit_xyz": np.asarray([[[1.0, 0.0, 0.0]]], dtype=float),
        "hit_alive": np.asarray([[True]], dtype=bool),
        "hit_time": np.asarray([0.0], dtype=float),
        "hit_phi_index": np.asarray([0], dtype=int),
        "stop_time": np.asarray([0.125], dtype=float),
        "stop_xyz": np.asarray([[1.0, 0.0, 0.0]], dtype=float),
        "stop_reason": np.asarray([module.PHI_ODE_SINGULARITY_STOP_REASON], dtype=int),
        "final_states": np.asarray([[1.0, 0.0]], dtype=float),
        "final_alive": np.asarray([False], dtype=bool),
        "final_phi": np.asarray(0.25, dtype=float),
    }
    fieldlines_tys, fieldlines_phi_hits, phis = (
        module.jax_trace_records_to_simsopt_format(trace_result, nfp=2)
    )
    metrics = trace_metrics(
        fieldlines_tys,
        fieldlines_phi_hits,
        phis,
        module.JAX_DEFAULT_STOP_LABELS,
        mode="default",
    )

    assert metrics["stop_reason_counts"]["phi_ode_singularity"] == 1
    assert metrics["stop_reason_counts"]["iteration_limit"] == 0


class _TraceDomain:
    def as_metadata(self):
        return {"rmin": 0.5, "rmax": 1.5, "zmax": 0.5}


def _runner_args():
    return types.SimpleNamespace(
        jax_platform="",
        no_vtk=True,
        nfieldlines=1,
        cpu_tmax_reference=7000.0,
        field_periods=1,
        steps_per_field_period=4,
        min_bphi_over_b=1e-10,
        dpi=40,
        tol_reference=1e-7,
    )


def _default_render_modes(*args, **kwargs):
    default_mode = {
        "mode": "default",
        "metrics_suffix": "_default",
        "plot_suffix": "_default",
        "seed_contract": {"kind": "extended_surface"},
        "trace_domain": _TraceDomain(),
        "stop_labels": module.JAX_DEFAULT_STOP_LABELS,
        "trace_semantics": "baseline_wander",
        "field_key": "baseline_wander",
        "radii": np.asarray([1.0]),
        "z0": np.asarray([0.0]),
    }
    return [
        {"mode": "validation", "metrics_suffix": "_validation"},
        {"mode": "diagnostic", "metrics_suffix": "_diagnostic"},
        default_mode,
    ], default_mode


def _patch_minimal_runner(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "configure_jax_platform", lambda platform: None)
    monkeypatch.setattr(module, "resolve_output_dir", lambda: tmp_path)
    monkeypatch.setattr(
        module,
        "load_field_and_surface",
        lambda out_dir: (
            object(),
            types.SimpleNamespace(
                nfp=2,
                quadpoints_phi=[0.0],
                quadpoints_theta=[0.0],
            ),
            "opt",
            {"FINAL_IOTA": 0.09},
            False,
        ),
    )
    monkeypatch.setattr(module, "build_default_render_modes", _default_render_modes)


def test_run_jax_default_poincare_rejects_stale_unselected_metrics(
    monkeypatch,
    tmp_path,
):
    _patch_minimal_runner(monkeypatch, tmp_path)
    (tmp_path / "PoincareMetrics_opt_validation.json").write_text(
        "{}",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "extract_discrete_biot_savart_coils",
        lambda bs: pytest.fail("stale metrics must fail before tracing"),
    )

    with pytest.raises(FileExistsError, match="stale metrics"):
        module.run_jax_default_poincare(_runner_args())


def test_run_jax_default_poincare_writes_canonical_default_artifacts(
    monkeypatch,
    tmp_path,
):
    _patch_minimal_runner(monkeypatch, tmp_path)
    coil_data = types.SimpleNamespace(
        ncoils=1,
        nquadpoints=4,
        quadrature_counts=np.asarray([4]),
    )
    monkeypatch.setattr(
        module,
        "extract_discrete_biot_savart_coils",
        lambda bs: coil_data,
    )
    monkeypatch.setattr(
        module,
        "trace_default_mode_jax",
        lambda *args, **kwargs: ([], [], [0.0, 0.25, 0.5, 0.75], 0.02),
    )
    monkeypatch.setattr(
        module,
        "_trace_metrics",
        lambda *args, **kwargs: {
            "per_phi_hit_counts": [0, 0, 0, 0],
            "validation_status": "diagnostic_only",
            "survived_lines": 0,
            "nfieldlines": 1,
        },
    )

    def touch_plot(fieldlines_phi_hits, phis, filename, **kwargs):
        with open(filename, "w", encoding="utf-8") as output_file:
            output_file.write("plot")

    monkeypatch.setattr(module, "plot_poincare_data", touch_plot)

    artifact = module.run_jax_default_poincare(_runner_args())
    metrics_path = tmp_path / "PoincareMetrics_opt_default.json"
    plot_path = tmp_path / "PoincarePlot_opt_default.png"

    assert metrics_path.exists()
    assert plot_path.exists()
    assert not (tmp_path / "PoincareMetrics_opt_default_jax.json").exists()
    assert artifact["field_model"]["poincare_field_key"] == "baseline_wander"
    assert artifact["field_model"]["poincare_trace_semantics"] == "baseline_wander"
    assert artifact["plot_filename"] == plot_path.name
