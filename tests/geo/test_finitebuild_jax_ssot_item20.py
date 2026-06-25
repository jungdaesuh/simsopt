"""SSOT analytic-oracle and runtime tests for the pure JAX finite-build kernel.

Wave R4 item 20 closure: ``simsopt_jax.core.finitebuild`` exposes the
kernel-level filament construction used by
``simsopt.geo.finitebuild.create_multifilament_grid``. The legacy
``CurveFilament`` class does not own immutable JAX specs or JAX graft
methods in the pure split, so this module constructs explicit specs for
the SSOT kernels and checks geometry against a hand-derived closed-form
planar-circle oracle.

Audit finding #3 reference:
``.artifacts/jax-test-audit-2026-05-13/TEST_QUALITY_TODOS.md``.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from simsopt.geo import FrameRotation, ZeroRotation
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.finitebuild import create_multifilament_grid
from simsopt_jax.core import (
    build_filament_gamma_and_dash,
    build_filament_gammas,
    compute_filament_offsets,
    make_curve_filament_spec,
    make_curve_xyzfourier_spec,
    make_frame_rotation_spec,
    make_optimizable_dof_map_spec,
    make_zero_rotation_spec,
)


_NQUADPOINTS = 64
_CURVE_ORDER = 3
_GRID_NN = 2
_GRID_NB = 3
_GAP_N = 0.025
_GAP_B = 0.018
_RNG_SEED = 4242
_PERTURB_SCALE = 0.02
_ANALYTIC_RTOL = 1e-12
_ANALYTIC_ATOL = 1e-12


def _unit_circle_curve(nquadpoints: int = _NQUADPOINTS) -> CurveXYZFourier:
    """Return a ``CurveXYZFourier`` parameterising the planar unit circle.

    The Fourier ordering for order 1 stores DOFs as
    ``[xc(0), xs(1), xc(1), yc(0), ys(1), yc(1), zc(0), zs(1), zc(1)]``,
    so setting ``xc(1) = 1`` and ``ys(1) = 1`` yields
    ``gamma(t) = (cos(2 pi t), sin(2 pi t), 0)``.
    """
    curve = CurveXYZFourier(nquadpoints, 1)
    dofs = np.zeros(9, dtype=np.float64)
    dofs[2] = 1.0  # xc(1)
    dofs[4] = 1.0  # ys(1)
    curve.x = dofs
    return curve


def _seed_curve() -> CurveXYZFourier:
    """Non-planar perturbed curve used by the offset and transfer-guard tests."""
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


def _local_dof_map(template_dofs: np.ndarray, owner_start: int):
    size = int(template_dofs.size)
    owner_segments = (
        ((int(owner_start), int(owner_start) + size, 0, size),) if size else ()
    )
    return make_optimizable_dof_map_spec(
        template_full_dofs=template_dofs,
        owner_segments=owner_segments,
        input_mode="local",
        input_start=0,
        input_end=size,
    )


def _rotation_spec_and_dofs(curve: CurveXYZFourier, rotation):
    if isinstance(rotation, ZeroRotation):
        return make_zero_rotation_spec(quadpoints=curve.quadpoints), np.empty(0)
    if isinstance(rotation, FrameRotation):
        dofs = np.asarray(rotation.full_x, dtype=np.float64)
        return (
            make_frame_rotation_spec(
                dofs=dofs,
                quadpoints=curve.quadpoints,
                order=rotation.order,
                scale=rotation.scale,
            ),
            dofs,
        )
    raise TypeError(f"Unsupported finite-build rotation: {type(rotation).__name__}")


def _curvefilament_spec(
    *,
    curve: CurveXYZFourier,
    rotation,
    frame_kind: str,
    dn: float,
    db: float,
    order: int,
):
    base_dofs = np.asarray(curve.get_dofs(), dtype=np.float64)
    rotation_spec, rotation_dofs = _rotation_spec_and_dofs(curve, rotation)
    owner_dofs = (
        np.concatenate((base_dofs, rotation_dofs))
        if rotation_dofs.size
        else base_dofs.copy()
    )
    return make_curve_filament_spec(
        dofs=owner_dofs,
        quadpoints=curve.quadpoints,
        base_curve=make_curve_xyzfourier_spec(
            dofs=base_dofs,
            quadpoints=curve.quadpoints,
            order=order,
        ),
        base_curve_map=_local_dof_map(base_dofs, 0),
        rotation=rotation_spec,
        rotation_map=_local_dof_map(rotation_dofs, base_dofs.size),
        frame_kind=frame_kind,
        dn=dn,
        db=db,
    )


def test_build_filament_gammas_matches_planar_circle_closed_form():
    """Closed-form planar-circle oracle pins the SSOT filament builder.

    For ``gamma(t) = (cos(2 pi t), sin(2 pi t), 0)`` on the planar unit
    circle, the unrotated centroid frame has
    ``T(t) = (-sin(2 pi t),  cos(2 pi t), 0)``,
    ``N(t) = ( cos(2 pi t),  sin(2 pi t), 0)``,
    ``B(t) = T x N = (0, 0, -1)``.
    With ``alpha = 0`` (``rotation_order=None`` => ``ZeroRotation``),
    the filament at offset ``(dn, db)`` is therefore

        gamma_fil(t)     = ((1 + dn) cos(2 pi t), (1 + dn) sin(2 pi t), -db)
        gammadash_fil(t) = 2 pi * (-(1 + dn) sin(2 pi t), (1 + dn) cos(2 pi t), 0).

    These formulas are independent of ``build_filament_gammas``,
    ``CurveFilament.gamma()``, and ``create_multifilament_grid``; agreement
    at ``rtol=1e-12`` therefore proves the SSOT kernel matches the
    Singh et al. (2020) construction up to floating-point round-off.
    """
    curve = _unit_circle_curve()
    # 1x1 grid yields a single centred filament that carries the
    # ``ZeroRotation`` frame; offsets are supplied explicitly below.
    filaments = create_multifilament_grid(
        curve,
        numfilaments_n=1,
        numfilaments_b=1,
        gapsize_n=1.0,
        gapsize_b=1.0,
        rotation_order=None,
        frame="centroid",
    )
    spec = _curvefilament_spec(
        curve=curve,
        rotation=filaments[0].rotation,
        frame_kind="centroid",
        dn=filaments[0].dn,
        db=filaments[0].db,
        order=1,
    )

    dn = 0.1
    db = 0.05
    filament_arrays = build_filament_gammas(spec, ((dn, db),))
    assert len(filament_arrays) == 1

    gamma_jax = np.asarray(filament_arrays[0][0], dtype=np.float64)
    gammadash_jax = np.asarray(filament_arrays[0][1], dtype=np.float64)

    two_pi_t = 2.0 * np.pi * np.asarray(curve.quadpoints, dtype=np.float64)
    cos_t = np.cos(two_pi_t)
    sin_t = np.sin(two_pi_t)
    z_offset = np.full_like(cos_t, -db)
    analytic_gamma = np.column_stack([(1.0 + dn) * cos_t, (1.0 + dn) * sin_t, z_offset])
    analytic_gammadash = np.column_stack(
        [
            -2.0 * np.pi * (1.0 + dn) * sin_t,
            2.0 * np.pi * (1.0 + dn) * cos_t,
            np.zeros_like(cos_t),
        ]
    )

    np.testing.assert_allclose(
        gamma_jax,
        analytic_gamma,
        rtol=_ANALYTIC_RTOL,
        atol=_ANALYTIC_ATOL,
    )
    np.testing.assert_allclose(
        gammadash_jax,
        analytic_gammadash,
        rtol=_ANALYTIC_RTOL,
        atol=_ANALYTIC_ATOL,
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


def test_curve_filament_spec_uses_base_curve_and_rotation_dofs():
    """CurveFilament specs carry parent-curve and rotation DOFs explicitly."""
    curve = _seed_curve()
    filaments = create_multifilament_grid(
        curve,
        numfilaments_n=2,
        numfilaments_b=2,
        gapsize_n=_GAP_N,
        gapsize_b=_GAP_B,
        rotation_order=1,
        frame="centroid",
    )
    filament = filaments[0]
    spec = _curvefilament_spec(
        curve=curve,
        rotation=filament.rotation,
        frame_kind="centroid",
        dn=filament.dn,
        db=filament.db,
        order=_CURVE_ORDER,
    )
    base_size = int(np.asarray(curve.get_dofs(), dtype=np.float64).size)
    rotation_size = int(np.asarray(filament.rotation.full_x, dtype=np.float64).size)

    assert filament.local_dof_size == 0
    assert filament.dof_size > 0
    assert spec.dofs.shape == (base_size + rotation_size,)
    assert spec.base_curve_map.owner_segments == ((0, base_size, 0, base_size),)
    assert spec.rotation_map.owner_segments == (
        (base_size, base_size + rotation_size, 0, rotation_size),
    )


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
    spec = jax.device_put(
        _curvefilament_spec(
            curve=curve,
            rotation=filaments[0].rotation,
            frame_kind="centroid",
            dn=filaments[0].dn,
            db=filaments[0].db,
            order=_CURVE_ORDER,
        )
    )
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
