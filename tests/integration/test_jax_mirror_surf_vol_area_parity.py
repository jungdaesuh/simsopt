"""Matched native/JAX parity for the exact ``surf_vol_area.py`` mirror."""

from __future__ import annotations

from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.cases import native_surf_vol_area
from examples.jax.parity.input_bundle import load_input_bundle
from scipy.optimize import OptimizeResult
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import LeastSquaresProblem

# A/reference-simple/runs/native-surf-vol-area/captured-natural-omp1/capture.json
# holds the official grid's counts (9 + 5 evaluations); the bounded scale solves
# the same workflow on a coarser grid, so only the two lanes are compared here.
OFFICIAL_RAW_STATUS = (
    "1 `gtol` termination condition is satisfied."
    " | 1 `gtol` termination condition is satisfied."
)


def test_exact_surf_vol_area_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-surf-vol-area")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    native_directory = tmp_path / "native"
    native_directory.mkdir()
    with chdir(native_directory):
        native = case.execute("native-cpu", bundle, arrays)

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = case.execute("jax-cpu", bundle, arrays)

    assert native.success is True
    assert jax.success is True
    # One stopping rule for both lanes: the official SciPy defaults.
    assert native.normalized_status == jax.normalized_status == "converged"
    assert native.raw_status == jax.raw_status == OFFICIAL_RAW_STATUS
    # Stop reason (above) and end point (below), not counters: both lanes run
    # ``jac="2-point"`` over independent residual implementations, so the
    # trust-region trial sequence is decided by rounding-level differences and
    # equal counts are a coincidence, not an invariant. Published instead.
    print(
        "native-surf-vol-area bounded counters (nfev, njev): "
        f"native={(native.nfev, native.njev)} jax={(jax.nfev, jax.njev)}"
    )
    assert native.scale == jax.scale == "bounded"
    assert native.input_fingerprint == jax.input_fingerprint
    assert native.configuration_fingerprint == jax.configuration_fingerprint
    assert native.completed_workflow_stages == jax.completed_workflow_stages
    assert set(native.values) == set(jax.values)

    for stage in ("first", "second"):
        for phase in ("initial", "final"):
            for observable in (
                "parameter_invariants",
                "area",
                "volume",
                "residual",
                "residual_jacobian_invariants",
                "objective_sum_squares",
                "objective_gradient_invariants",
            ):
                name = f"{stage}:{phase}:{observable}"
                np.testing.assert_allclose(
                    jax.values[name],
                    native.values[name],
                    rtol=1.0e-8,
                    atol=1.0e-10,
                )

    np.testing.assert_allclose(
        native.values["first:final:residual"],
        np.zeros(2),
        rtol=0.0,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        native.values["second:final:residual"],
        np.zeros(2),
        rtol=0.0,
        atol=1.0e-8,
    )


def test_native_default_surf_vol_area_uses_native_implicit_grid(tmp_path: Path) -> None:
    """An unconfigured native surface is the official example's grid oracle."""
    native_surface = SurfaceRZFourier(mpol=1, ntor=0)
    case = get_case("native-surf-vol-area")
    bundle = case.create_input(tmp_path / "shipped", "native_default")
    _, arrays = load_input_bundle(tmp_path / "shipped", bundle)
    np.testing.assert_array_equal(arrays["quadrature"], native_surface.quadpoints_phi)
    np.testing.assert_array_equal(
        arrays["quadrature_theta"], native_surface.quadpoints_theta
    )
    bounded = case.create_input(tmp_path / "bounded", "bounded")
    _, bounded_arrays = load_input_bundle(tmp_path / "bounded", bounded)
    assert bounded_arrays["quadrature"].shape == (32,)
    assert "quadrature_theta" not in bounded_arrays


def test_native_default_surf_vol_area_uses_shipped_solver_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-surf-vol-area")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "native_default")
    _, arrays = load_input_bundle(input_root, bundle)

    calls: list[dict[str, float | int | str]] = []
    original_solve = native_surf_vol_area.solve_official_least_squares

    def recording_solve(
        problem: LeastSquaresProblem, **kwargs: float | int | str
    ) -> OptimizeResult:
        calls.append(kwargs)
        return original_solve(problem, **kwargs)

    monkeypatch.setattr(
        native_surf_vol_area, "solve_official_least_squares", recording_solve
    )
    native_directory = tmp_path / "native"
    native_directory.mkdir()
    with chdir(native_directory):
        native = case.execute("native-cpu", bundle, arrays)

    assert calls == [{}, {"diff_method": "centered"}]
    assert native.success is True
    for stage in ("first", "second"):
        np.testing.assert_allclose(
            native.values[f"{stage}:final:residual"],
            np.zeros(2),
            rtol=0.0,
            atol=1.0e-8,
        )


@pytest.mark.parametrize("preexisting", (False, True))
def test_surf_vol_area_roundtrip_preserves_cwd_surface_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preexisting: bool,
) -> None:
    """Neither lane creates or overwrites the cwd's serialization file."""
    case = get_case("native-surf-vol-area")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)
    cwd_directory = tmp_path / "cwd"
    cwd_directory.mkdir()
    sentinel = cwd_directory / "surf_fw.json"
    if preexisting:
        sentinel.write_bytes(b"sentinel-bytes")
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    with chdir(cwd_directory):
        for lane in ("native-cpu", "jax-cpu"):
            result = case.execute(lane, bundle, arrays)
            assert result.success is True
            assert result.normalized_status == "converged"
            assert sentinel.exists() is preexisting
            if preexisting:
                assert sentinel.read_bytes() == b"sentinel-bytes"
