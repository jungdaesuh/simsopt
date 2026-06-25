import types
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from examples.single_stage_optimization.POINCARE_PLOTTING import (
    poincare_surfaces_jax_default as module,
)


def _circular_loop(radius, nquad):
    t = np.linspace(0.0, 1.0, int(nquad), endpoint=False)
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
    return gamma, gammadash


def _axis_loop_bz(radius, current, z):
    return (
        4.0e-7
        * np.pi
        * current
        * radius**2
        / (2.0 * (radius**2 + z**2) ** 1.5)
    )


def test_biot_savart_jax_matches_circular_loop_axis_field():
    radius = 1.7
    current = 2.3
    z = 0.4
    nquad = 4096
    gamma, gammadash = _circular_loop(radius, nquad)

    field = module._biot_savart_points_jax(
        jnp.asarray([[0.0, 0.0, z]], dtype=jnp.float64),
        jnp.asarray(gamma[None, :, :], dtype=jnp.float64),
        jnp.asarray(gammadash[None, :, :], dtype=jnp.float64),
        jnp.ones((1, nquad), dtype=jnp.float64),
        jnp.asarray([current], dtype=jnp.float64),
    )

    np.testing.assert_allclose(
        np.asarray(field)[0],
        [0.0, 0.0, _axis_loop_bz(radius, current, z)],
        rtol=1e-5,
        atol=1e-12,
    )


def test_biot_savart_jax_ignores_masked_padding_samples():
    radius = 1.2
    current = 1.4
    z = 0.3
    nquad = 128
    padded_nquad = 256
    gamma, gammadash = _circular_loop(radius, nquad)
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

    np.testing.assert_allclose(
        np.asarray(padded)[0],
        [0.0, 0.0, _axis_loop_bz(radius, current, z)],
        rtol=1e-5,
        atol=1e-12,
    )


def test_jax_trace_settings_require_four_plane_alignment():
    assert module.validate_jax_trace_settings(3, 64) == 16
    with pytest.raises(ValueError, match="divisible by 4"):
        module.validate_jax_trace_settings(3, 62)


def test_jax_trace_records_preserve_simsopt_plot_contract():
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
    metrics = module.trace_metrics(
        fieldlines_phi_hits,
        trace_result,
        phis,
        nfieldlines=2,
    )

    assert fieldlines_tys[1][0, 0] == pytest.approx(0.4)
    assert fieldlines_phi_hits[1][-1, 1] == -1
    assert metrics["survived_lines"] == 1
    assert metrics["lost_lines"] == 1
    assert metrics["stop_reason_counts"]["max_z_guardrail"] == 1


def test_jax_invalid_field_stop_is_not_reported_as_box_guardrail():
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


def _runner_args():
    return types.SimpleNamespace(
        jax_platform="",
        no_vtk=True,
        use_opt=False,
        nfieldlines=1,
        cpu_tmax_reference=7000.0,
        field_periods=1,
        steps_per_field_period=4,
        min_bphi_over_b=1e-10,
        dpi=40,
        tol_reference=1e-7,
        extend_distance=0.05,
    )


def _touch_poincare_inputs(out_dir):
    for filename in (
        "biot_savart_init.json",
        "surf_init.json",
        "biot_savart_opt.json",
        "surf_opt.json",
    ):
        (out_dir / filename).write_text("{}", encoding="utf-8")


def test_load_field_and_surface_defaults_to_cpu_init_pair(monkeypatch, tmp_path):
    """Default JAX mode must load the same input pair as poincare_surfaces.py."""
    _touch_poincare_inputs(tmp_path)
    loaded = []

    def fake_load(filename):
        loaded.append(Path(filename).name)
        return types.SimpleNamespace()

    monkeypatch.setattr(module, "load", fake_load)

    _, _, field_label, surface_path = module.load_field_and_surface(tmp_path)

    assert loaded == ["biot_savart_init.json", "surf_init.json"]
    assert field_label == "init"
    assert surface_path == tmp_path / "surf_init.json"


def test_load_field_and_surface_uses_opt_pair_only_when_requested(
    monkeypatch,
    tmp_path,
):
    _touch_poincare_inputs(tmp_path)
    loaded = []

    def fake_load(filename):
        loaded.append(Path(filename).name)
        return types.SimpleNamespace()

    monkeypatch.setattr(module, "load", fake_load)

    _, _, field_label, surface_path = module.load_field_and_surface(
        tmp_path,
        use_opt=True,
    )

    assert loaded == ["biot_savart_opt.json", "surf_opt.json"]
    assert field_label == "opt"
    assert surface_path == tmp_path / "surf_opt.json"


def test_use_opt_requires_complete_optimized_input(tmp_path):
    (tmp_path / "biot_savart_opt.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="--use-opt requested"):
        module.load_field_and_surface(tmp_path, use_opt=True)


def test_run_jax_default_poincare_writes_jax_artifacts_without_cpu_overwrite(
    monkeypatch,
    tmp_path,
):
    existing_cpu_plot = tmp_path / "PoincarePlot.png"
    existing_cpu_plot.write_text("cpu-reference", encoding="utf-8")

    monkeypatch.setattr(module, "configure_jax_platform", lambda platform: None)
    monkeypatch.setattr(module, "resolve_output_dir", lambda: tmp_path)
    def fake_load_field_and_surface(out_dir, *, use_opt):
        assert use_opt is False
        return (
            object(),
            types.SimpleNamespace(
                nfp=2,
                quadpoints_phi=[0.0],
                quadpoints_theta=[0.0],
            ),
            "init",
            tmp_path / "surf_init.json",
        )

    monkeypatch.setattr(module, "load_field_and_surface", fake_load_field_and_surface)
    monkeypatch.setattr(
        module,
        "build_default_trace_domain",
        lambda surface_path, extend_distance: module.DefaultTraceDomain(
            seed_rmin=1.0,
            seed_rmax=1.0,
            stopping_rmin=0.5,
            stopping_rmax=1.5,
            stopping_zmax=0.5,
        ),
    )
    monkeypatch.setattr(
        module,
        "extract_discrete_biot_savart_coils",
        lambda bs: types.SimpleNamespace(
            ncoils=1,
            nquadpoints=4,
            quadrature_counts=np.asarray([4]),
        ),
    )
    monkeypatch.setattr(
        module,
        "trace_default_mode_jax",
        lambda *args, **kwargs: (
            [],
            [],
            [0.0, 0.25, 0.5, 0.75],
            {
                "stop_reason": np.asarray([-1], dtype=int),
                "final_alive": np.asarray([True], dtype=bool),
            },
            0.02,
        ),
    )
    monkeypatch.setattr(module, "trace_metrics", lambda *args, **kwargs: {
        "per_phi_hit_counts": [0, 0, 0, 0],
        "validation_status": "diagnostic_only",
        "survived_lines": 1,
        "lost_lines": 0,
        "nfieldlines": 1,
        "stop_reason_counts": {},
    })
    monkeypatch.setattr(
        module,
        "build_field_model_metadata",
        lambda *args, **kwargs: {"jax_backend": "cpu"},
    )

    def touch_plot(fieldlines_phi_hits, phis, filename, **kwargs):
        with open(filename, "w", encoding="utf-8") as output_file:
            output_file.write("jax-plot")

    monkeypatch.setattr(module, "plot_poincare_data", touch_plot)

    artifact = module.run_jax_default_poincare(_runner_args())

    assert existing_cpu_plot.read_text(encoding="utf-8") == "cpu-reference"
    assert (tmp_path / "PoincarePlot_init_default_jax.png").exists()
    assert (tmp_path / "PoincareMetrics_init_default_jax.json").exists()
    assert artifact["plot_filename"] == "PoincarePlot_init_default_jax.png"
    assert artifact["use_optimized_input"] is False
    assert artifact["cpu_default_reference"]["output"] == "PoincarePlot.png"
