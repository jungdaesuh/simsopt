"""SSOT parity tests for the pure JAX finite-build kernel.

Wave R4 item 20 closure: ``simsopt.jax_core.finitebuild`` exposes the
kernel-level filament construction used by
``simsopt.geo.finitebuild.create_multifilament_grid``. These tests pin
``build_filament_gammas`` to bit-identity with the host adapter on a
representative ``CurveXYZFourier + FrameRotation`` configuration at the
``direct_kernel`` parity-ladder lane.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.finitebuild import create_multifilament_grid
from simsopt.jax_core import (
    build_filament_gamma_and_dash,
    build_filament_gammas,
    compute_filament_offsets,
)


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]

_NQUADPOINTS = 64
_CURVE_ORDER = 3
_GRID_NN = 2
_GRID_NB = 3
_GAP_N = 0.025
_GAP_B = 0.018
_ROTATION_ORDER = 1
_RNG_SEED = 4242
_PERTURB_SCALE = 0.02


def _seed_curve() -> CurveXYZFourier:
    rng = np.random.default_rng(_RNG_SEED)
    curve = CurveXYZFourier(_NQUADPOINTS, _CURVE_ORDER)
    ndofs = 3 * (2 * _CURVE_ORDER + 1)
    dofs = np.zeros(ndofs, dtype=np.float64)
    dofs[2] = 1.0  # xc(1)
    dofs[(2 * _CURVE_ORDER + 1) + 1] = 1.0  # ys(1)
    dofs[2 * (2 * _CURVE_ORDER + 1) + 2] = 0.25  # zc(1) non-planar bump
    dofs += _PERTURB_SCALE * rng.standard_normal(ndofs)
    curve.x = dofs
    return curve


@pytest.mark.parametrize("frame_kind", ("centroid", "frenet"))
@pytest.mark.parametrize("rotation_order", (None, _ROTATION_ORDER))
def test_build_filament_gammas_matches_create_multifilament_grid(
    frame_kind: str,
    rotation_order: int | None,
):
    curve = _seed_curve()
    filaments = create_multifilament_grid(
        curve,
        numfilaments_n=_GRID_NN,
        numfilaments_b=_GRID_NB,
        gapsize_n=_GAP_N,
        gapsize_b=_GAP_B,
        rotation_order=rotation_order,
        frame=frame_kind,
    )

    if rotation_order is not None:
        shared_rotation = filaments[0].rotation
        rng = np.random.default_rng(_RNG_SEED + 1)
        shared_rotation.x = 0.1 * rng.standard_normal(shared_rotation.dof_size)

    offsets = compute_filament_offsets(
        numfilaments_n=_GRID_NN,
        numfilaments_b=_GRID_NB,
        gapsize_n=_GAP_N,
        gapsize_b=_GAP_B,
    )
    assert len(offsets) == len(filaments)
    for filament, (dn, db) in zip(filaments, offsets, strict=True):
        assert filament.dn == pytest.approx(dn)
        assert filament.db == pytest.approx(db)

    # Build one spec from the first filament; all filaments share the
    # base curve + rotation Optimizable graph, so one spec captures the
    # shared frame state.
    sample_spec = filaments[0].to_spec()

    filament_arrays = build_filament_gammas(sample_spec, offsets)
    assert len(filament_arrays) == len(filaments)

    for index, (filament, (gamma_jax, gammadash_jax)) in enumerate(
        zip(filaments, filament_arrays, strict=True)
    ):
        gamma_host = np.asarray(filament.gamma(), dtype=np.float64)
        gammadash_host = np.asarray(filament.gammadash(), dtype=np.float64)
        np.testing.assert_allclose(
            np.asarray(gamma_jax, dtype=np.float64),
            gamma_host,
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=f"SSOT gamma diverges for filament {index} ({frame_kind})",
        )
        np.testing.assert_allclose(
            np.asarray(gammadash_jax, dtype=np.float64),
            gammadash_host,
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=f"SSOT gammadash diverges for filament {index} ({frame_kind})",
        )


@pytest.mark.parametrize("frame_kind", ("centroid", "frenet"))
def test_build_filament_gamma_and_dash_matches_grid_first_filament(frame_kind: str):
    """Single-filament SSOT path matches the host adapter for offset (dn, db)."""
    curve = _seed_curve()
    filaments = create_multifilament_grid(
        curve,
        numfilaments_n=_GRID_NN,
        numfilaments_b=_GRID_NB,
        gapsize_n=_GAP_N,
        gapsize_b=_GAP_B,
        rotation_order=_ROTATION_ORDER,
        frame=frame_kind,
    )
    rng = np.random.default_rng(_RNG_SEED + 2)
    filaments[0].rotation.x = 0.1 * rng.standard_normal(filaments[0].rotation.dof_size)

    target = filaments[3]
    spec = target.to_spec()
    gamma_jax, gammadash_jax = build_filament_gamma_and_dash(spec)
    np.testing.assert_allclose(
        np.asarray(gamma_jax, dtype=np.float64),
        np.asarray(target.gamma(), dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(gammadash_jax, dtype=np.float64),
        np.asarray(target.gammadash(), dtype=np.float64),
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_compute_filament_offsets_matches_grid_construction():
    """Pure offset helper matches the host implementation for odd/even grids."""
    for nn, nb in ((3, 4), (1, 1), (2, 1), (4, 2)):
        gap_n = 0.011
        gap_b = 0.017
        offsets = compute_filament_offsets(
            numfilaments_n=nn,
            numfilaments_b=nb,
            gapsize_n=gap_n,
            gapsize_b=gap_b,
        )
        curve = _seed_curve()
        filaments = create_multifilament_grid(
            curve,
            numfilaments_n=nn,
            numfilaments_b=nb,
            gapsize_n=gap_n,
            gapsize_b=gap_b,
            rotation_order=None,
            frame="centroid",
        )
        assert len(offsets) == len(filaments)
        for filament, (dn, db) in zip(filaments, offsets, strict=True):
            assert filament.dn == pytest.approx(dn)
            assert filament.db == pytest.approx(db)


def test_compiled_filament_builders_run_under_strict_transfer_guard():
    """Compiled SSOT kernels execute without implicit host/device transfers."""
    curve = _seed_curve()
    filaments = create_multifilament_grid(
        curve,
        numfilaments_n=_GRID_NN,
        numfilaments_b=_GRID_NB,
        gapsize_n=_GAP_N,
        gapsize_b=_GAP_B,
        rotation_order=None,
        frame="centroid",
    )
    spec = jax.device_put(filaments[0].to_spec())
    offsets = compute_filament_offsets(
        numfilaments_n=_GRID_NN,
        numfilaments_b=_GRID_NB,
        gapsize_n=_GAP_N,
        gapsize_b=_GAP_B,
    )

    @jax.jit
    def build_one(spec_arg):
        return build_filament_gamma_and_dash(spec_arg)

    @jax.jit
    def build_all(spec_arg):
        return build_filament_gammas(spec_arg, offsets)

    build_one_compiled = build_one.lower(spec).compile()
    build_all_compiled = build_all.lower(spec).compile()

    with jax.transfer_guard("disallow"):
        gamma, gammadash = build_one_compiled(spec)
        filament_arrays = build_all_compiled(spec)

    assert gamma.shape == (_NQUADPOINTS, 3)
    assert gammadash.shape == (_NQUADPOINTS, 3)
    assert len(filament_arrays) == _GRID_NN * _GRID_NB
    for filament_gamma, filament_gammadash in filament_arrays:
        assert filament_gamma.shape == (_NQUADPOINTS, 3)
        assert filament_gammadash.shape == (_NQUADPOINTS, 3)
