"""Native-vs-JAX parity for the ``1_Simple/permanent_magnet_simple.py`` mirror.

Both lanes solve the same live grid: native ``simsopt.solve.GPMO`` (upstream's
baseline algorithm) and ``GPMO_baseline_jax`` on that grid staged with
``PermanentMagnetGridJAX.from_cpu``. The shipped mirror script is executed as a
program (its directory is not an importable package) and its published result
is read back from ``--json``.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import hashlib
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import jax
import numpy as np
import pytest
from simsopt.field import ToroidalField
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.solve import GPMO
from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
from simsopt_jax.solve.permanent_magnet import GPMO_baseline_jax
from simsopt_jax_adapters.isolated_kernel import repo_child_pythonpath

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA = REPOSITORY_ROOT / "tests" / "test_files"
MIRROR_SOURCE = (
    REPOSITORY_ROOT / "examples" / "jax" / "1_Simple" / "permanent_magnet_simple.py"
)
NET_POLOIDAL_CURRENT_AMPERES = 3.7713e6
VACUUM_PERMEABILITY = 4.0 * np.pi * 1.0e-7
#: Upstream's ``in_github_actions`` configuration of the simple script.
BOUNDED_RESOLUTION = 2
BOUNDED_DOWNSAMPLE = 100
BOUNDED_ITERATIONS = 40
#: Upstream's shipped resolution of the simple script.
NATIVE_RESOLUTION = 16
NATIVE_DOWNSAMPLE = 4
HISTORY_COUNT = 10
SINGLE_DIRECTION = -1
#: The JAX lane's runtime: the parity backend in FP64.
JAX_LANE_ENVIRONMENT = {
    "SIMSOPT_BACKEND_MODE": "jax_cpu_parity",
    "SIMSOPT_PRECISION": "fp64",
    "JAX_ENABLE_X64": "1",
}


def _simple_grid(resolution: int, downsample: int) -> PermanentMagnetGrid:
    """The cylindrical FAMUS grid of the simple example."""
    surface = SurfaceRZFourier.from_wout(
        str(TEST_DATA / "wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc"),
        range="half period",
        nphi=resolution,
        ntheta=resolution,
    )
    field = ToroidalField(
        R0=1.0,
        B0=VACUUM_PERMEABILITY * NET_POLOIDAL_CURRENT_AMPERES / (2.0 * np.pi),
    )
    field.set_points(surface.gamma().reshape((-1, 3)))
    normal_field = np.sum(
        field.B().reshape((resolution, resolution, 3)) * surface.unitnormal(),
        axis=2,
    )
    with redirect_stdout(io.StringIO()):
        return PermanentMagnetGrid.geo_setup_from_famus(
            surface,
            normal_field,
            TEST_DATA / "init_orient_pm_nonorm_5E4_q4_dp.focus",
            coordinate_flag="cylindrical",
            downsample=downsample,
        )


def _values(
    *,
    response: np.ndarray,
    target: np.ndarray,
    moment_maxima: np.ndarray,
    dipole_grid_xyz: np.ndarray,
    initial_moments: np.ndarray,
    final_moments: np.ndarray,
    final_residual: np.ndarray,
) -> dict[str, np.ndarray]:
    """One lane's construction, start state and endpoint, by observable name."""
    initial_residual = response @ initial_moments.reshape(-1) - target
    nonzero_mask = np.linalg.norm(final_moments, axis=1) != 0.0
    return {
        "construction:response_matrix": response,
        "construction:target": target,
        "construction:moment_maxima": moment_maxima,
        "construction:dipole_grid_xyz": dipole_grid_xyz,
        "initial:moments": initial_moments,
        "initial:residual": initial_residual,
        "initial:objective_sum_squares": np.asarray(
            np.vdot(initial_residual, initial_residual), dtype=np.float64
        ),
        "final:moments": final_moments,
        "final:residual": final_residual,
        "final:objective_sum_squares": np.asarray(
            np.vdot(final_residual, final_residual), dtype=np.float64
        ),
        "final:nonzero_mask": nonzero_mask,
        "final:nonzero_fraction": np.asarray(
            np.count_nonzero(nonzero_mask) / nonzero_mask.size, dtype=np.float64
        ),
    }


def _run_mirror(*arguments: str) -> dict[str, object]:
    """Execute the shipped mirror in the JAX lane and return its published result."""
    completed = subprocess.run(
        (sys.executable, "-S", str(MIRROR_SOURCE), "--json", *arguments),
        cwd=REPOSITORY_ROOT,
        # ``-S`` drops site-packages: the sources, the loaded kernel and the
        # dependency root are passed explicitly.
        env={
            **os.environ,
            **JAX_LANE_ENVIRONMENT,
            "PYTHONPATH": repo_child_pythonpath(
                REPOSITORY_ROOT, os.environ.get("PYTHONPATH")
            ),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    published = json.loads(completed.stdout.strip().splitlines()[-1])
    assert isinstance(published, dict)
    return published


def test_exact_permanent_magnet_simple_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = _simple_grid(BOUNDED_RESOLUTION, BOUNDED_DOWNSAMPLE)
    for name, value in JAX_LANE_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    # Staged before the native solve rebinds ``grid.m``: both lanes start from
    # the same construction.
    staged = PermanentMagnetGridJAX.from_cpu(grid)
    response = np.array(grid.A_obj, dtype=np.float64, copy=True)
    target = np.array(grid.b_obj, dtype=np.float64, copy=True)
    maxima = np.array(grid.m_maxima, dtype=np.float64, copy=True).reshape(-1)
    dipole_grid_xyz = np.array(grid.dipole_grid_xyz, dtype=np.float64, copy=True)
    initial_moments = np.array(grid.m0, dtype=np.float64, copy=True).reshape((-1, 3))

    with redirect_stdout(io.StringIO()):
        GPMO(
            grid,
            "baseline",
            K=BOUNDED_ITERATIONS,
            nhistory=HISTORY_COUNT,
            single_direction=SINGLE_DIRECTION,
            verbose=False,
        )
    native_moments = np.asarray(grid.m, dtype=np.float64).reshape((-1, 3))
    native = _values(
        response=response,
        target=target,
        moment_maxima=maxima,
        dipole_grid_xyz=dipole_grid_xyz,
        initial_moments=initial_moments,
        final_moments=native_moments,
        final_residual=response @ native_moments.reshape(-1) - target,
    )

    device_result = GPMO_baseline_jax(
        staged,
        K=BOUNDED_ITERATIONS,
        reg_l2=0.0,
        single_direction=SINGLE_DIRECTION,
        retain_history=False,
    )
    host_result, jax_response, jax_target, jax_maxima, jax_xyz, jax_initial = (
        jax.device_get(
            (
                device_result,
                staged.A_obj,
                staged.b_obj,
                staged.m_maxima,
                staged.dipole_grid_xyz,
                staged.m0,
            )
        )
    )
    jax_values = _values(
        response=np.asarray(jax_response, dtype=np.float64),
        target=np.asarray(jax_target, dtype=np.float64),
        moment_maxima=np.asarray(jax_maxima, dtype=np.float64),
        dipole_grid_xyz=np.asarray(jax_xyz, dtype=np.float64),
        initial_moments=np.asarray(jax_initial, dtype=np.float64),
        final_moments=np.asarray(host_result.m, dtype=np.float64),
        final_residual=np.asarray(host_result.residual, dtype=np.float64),
    )

    assert set(native) == set(jax_values)
    for observable in native:
        np.testing.assert_allclose(
            jax_values[observable],
            native[observable],
            rtol=1.0e-13,
            atol=1.0e-14,
            err_msg=observable,
        )

    assert native["construction:response_matrix"].shape == (4, 1722)
    assert native["construction:dipole_grid_xyz"].shape == (574, 3)
    assert int(np.count_nonzero(native["final:nonzero_mask"])) == 40
    assert float(native["final:objective_sum_squares"]) < float(
        native["initial:objective_sum_squares"]
    )

    # The mirror publishes the placed rows, not the 574-row moment array: the
    # bounded solve leaves 534 of those rows exactly zero. Its published rows
    # must still carry the cross-lane values asserted above, and its digest
    # must be the digest of the full array those rows scatter back into.
    bounded = _run_mirror("--smoke", "--output-dir", str(tmp_path))
    assert bounded["status"] == "ok"
    observables = bounded["observables"]
    assert isinstance(observables, dict)
    indices = np.asarray(observables["selected_moment_indices"], dtype=np.int64)
    selected_moments = np.asarray(observables["selected_moments"], dtype=np.float64)
    ndipoles = int(native["construction:dipole_grid_xyz"].shape[0])

    assert observables["ndipoles"] == ndipoles
    selected_count = int(np.count_nonzero(native["final:nonzero_mask"]))
    assert observables["selected_moment_count"] == selected_count
    assert indices.shape == (selected_count,)
    assert selected_moments.shape == (selected_count, 3)
    np.testing.assert_allclose(
        selected_moments,
        native["final:moments"][indices],
        rtol=1.0e-13,
        atol=1.0e-14,
    )

    scattered = np.zeros((ndipoles, 3), dtype=np.float64)
    scattered[indices] = selected_moments
    assert (
        hashlib.sha256(scattered.tobytes()).hexdigest() == observables["moments_sha256"]
    )


def test_mirror_solve_plumbs_native_default_scale_to_the_grid(
    tmp_path: Path,
) -> None:
    """``solve`` honours its scale argument: 16x16 rows, downsample-4 dipoles."""
    steps = 2
    reference = _simple_grid(NATIVE_RESOLUTION, NATIVE_DOWNSAMPLE)

    result = _run_mirror("--max-steps", str(steps), "--output-dir", str(tmp_path))
    observables = result["observables"]
    assert isinstance(observables, dict)

    assert result["scale"] == "native_default"
    assert observables["ndipoles"] == int(reference.ndipoles)
    assert observables["ndipoles"] > 10_000
    assert observables["selected_moment_count"] == steps
    assert len(observables["selected_moments"]) == steps
    assert len(observables["selected_moment_indices"]) == steps
    assert len(observables["selected_dipoles"]) == steps
    assert len(str(observables["moments_sha256"])) == 64
    assert result["status"] == "ok"
