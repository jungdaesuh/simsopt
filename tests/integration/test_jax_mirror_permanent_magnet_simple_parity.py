"""Exact parity for the ``1_Simple/permanent_magnet_simple.py`` mirror."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.cases.native_permanent_magnet_simple import (
    _build_cpu_grid,
    _scale_configuration,
)
from examples.jax.parity.input_bundle import load_input_bundle
from simsopt_jax.examples import ExecutionScale

MIRROR_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "jax"
    / "1_Simple"
    / "permanent_magnet_simple.py"
)


def _mirror() -> ModuleType:
    """Load the shipped mirror the way a reader would run it.

    Example scripts live outside any importable package, so this mirrors the
    loader the neighbouring example tests already use.
    """
    specification = importlib.util.spec_from_file_location(
        "jax_example_permanent_magnet_simple",
        MIRROR_SOURCE,
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_exact_permanent_magnet_simple_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-permanent-magnet-simple")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    native = case.execute("native-cpu", bundle, arrays)

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = case.execute("jax-cpu", bundle, arrays)

    assert native.success is True
    assert jax.success is True
    assert native.scale == jax.scale == "bounded"
    assert native.input_fingerprint == jax.input_fingerprint
    assert native.configuration_fingerprint == jax.configuration_fingerprint
    assert native.effective_construction_fingerprint == (
        jax.effective_construction_fingerprint
    )
    assert native.completed_workflow_stages == jax.completed_workflow_stages
    assert native.nit == jax.nit == 40
    assert set(native.values) == set(jax.values)

    for observable in (
        "construction:response_matrix",
        "construction:target",
        "construction:moment_maxima",
        "construction:dipole_grid_xyz",
        "initial:moments",
        "initial:residual",
        "initial:objective_sum_squares",
        "final:moments",
        "final:residual",
        "final:objective_sum_squares",
        "final:nonzero_mask",
        "final:nonzero_fraction",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-13,
            atol=1.0e-14,
        )

    assert native.values["construction:response_matrix"].shape == (4, 1722)
    assert native.values["construction:dipole_grid_xyz"].shape == (574, 3)
    assert int(np.count_nonzero(native.values["final:nonzero_mask"])) == 40
    assert float(native.values["final:objective_sum_squares"]) < float(
        native.values["initial:objective_sum_squares"]
    )

    # The mirror publishes the placed rows, not the 574-row moment array: the
    # bounded solve leaves 534 of those rows exactly zero. Its published rows
    # must still carry the cross-lane values asserted above, and its digest
    # must be the digest of the full array those rows scatter back into.
    mirror = _mirror()
    bounded = mirror.solve(tmp_path, mirror.BOUNDED_ITERATIONS, "bounded")
    observables = bounded.observables
    indices = np.asarray(observables["selected_moment_indices"], dtype=np.int64)
    selected_moments = np.asarray(observables["selected_moments"], dtype=np.float64)
    ndipoles = int(native.values["construction:dipole_grid_xyz"].shape[0])

    assert observables["ndipoles"] == ndipoles
    assert observables["selected_moment_count"] == native.nit
    assert indices.shape == (native.nit,)
    assert selected_moments.shape == (native.nit, 3)
    np.testing.assert_allclose(
        selected_moments,
        native.values["final:moments"][indices],
        rtol=1.0e-13,
        atol=1.0e-14,
    )

    scattered = np.zeros((ndipoles, 3), dtype=np.float64)
    scattered[indices] = selected_moments
    assert (
        hashlib.sha256(scattered.tobytes()).hexdigest() == observables["moments_sha256"]
    )


@pytest.mark.parametrize("scale", ("bounded", "native_default"))
def test_mirror_grid_follows_the_parity_case_scale_configuration(
    scale: ExecutionScale,
) -> None:
    """The mirror's grid is the one its parity case freezes for that scale."""
    configuration = _scale_configuration(scale)
    reference = _build_cpu_grid(configuration)

    grid = _mirror()._build_grid(scale)

    assert grid.nphi == configuration["nphi"]
    assert grid.ntheta == configuration["ntheta"]
    assert grid.b_obj.shape == (grid.nphi * grid.ntheta,)
    assert grid.ndipoles == int(reference.ndipoles)
    # Values, not just shapes. ``A_obj`` is 88 MB at native_default, so the
    # target field and the dipole moment maxima carry the equality: together
    # they pin the reduction rows and the whole dipole inventory the greedy
    # step ranks over.
    assert np.array_equal(np.asarray(grid.b_obj), reference.b_obj)
    assert np.array_equal(
        np.asarray(grid.m_maxima),
        np.asarray(reference.m_maxima).reshape((int(reference.ndipoles),)),
    )


def test_mirror_solve_plumbs_native_default_scale_to_the_grid(
    tmp_path: Path,
) -> None:
    """``solve`` honours its scale argument: 16x16 rows, downsample-4 dipoles."""
    steps = 2
    reference = _build_cpu_grid(_scale_configuration("native_default"))
    mirror = _mirror()

    result = mirror.solve(tmp_path, steps, "native_default")

    # The mirror's iteration budgets are the parity case's, not a second
    # independent literal.
    assert (
        mirror.NATIVE_ITERATIONS == _scale_configuration("native_default")["iterations"]
    )
    assert mirror.BOUNDED_ITERATIONS == _scale_configuration("bounded")["iterations"]

    assert result.observables["ndipoles"] == int(reference.ndipoles)
    assert result.observables["ndipoles"] > 10_000
    assert result.observables["selected_moment_count"] == steps
    assert len(result.observables["selected_moments"]) == steps
    assert len(result.observables["selected_moment_indices"]) == steps
    assert len(result.observables["selected_dipoles"]) == steps
    assert len(str(result.observables["moments_sha256"])) == 64
    assert result.status == "ok"
