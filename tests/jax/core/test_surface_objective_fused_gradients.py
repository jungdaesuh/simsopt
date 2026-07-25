"""Strict-placement coverage for fused surface-objective differentiation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from conftest import enable_strict_parity_backend, parity_default_device
from simsopt.field.coil import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier, Volume
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt_jax.core.surface_fourier_kernels import dofs_to_xyzc
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
import simsopt_jax_adapters.geo.surface_objectives as surface_objectives_module
from simsopt_jax_adapters.geo.surface_objectives import (
    BoozerResidualJAX,
    NonQuasiSymmetricRatioJAX,
    _make_cached_strict_scalar_value_and_two_gradients,
)


def test_fused_direct_and_inner_gradients_obey_strict_gpu_transfer_guard(
    monkeypatch,
    request,
) -> None:
    enable_strict_parity_backend(monkeypatch, request, "gpu")
    scatter_indices = np.asarray((0, 8, 9, 17, 18, 26), dtype=np.int32)

    def objective(coil_dofs, inner_dofs, _optimize_G, _weight_inv_modB):
        xc, yc, zc = dofs_to_xyzc(inner_dofs, scatter_indices, 1, 1)
        return jnp.sum(coil_dofs) + jnp.sum(xc) + jnp.sum(yc) + jnp.sum(zc)

    value_and_gradients = _make_cached_strict_scalar_value_and_two_gradients(
        objective
    )
    with parity_default_device("gpu"):
        device = jax.devices("gpu")[0]
        coil_dofs = jax.device_put(
            np.asarray((0.25, -0.5), dtype=np.float64),
            device,
        )
        inner_dofs = jax.device_put(
            np.asarray((1.0, -2.0, 3.0, -4.0, 5.0, -6.0), dtype=np.float64),
            device,
        )

        with jax.transfer_guard("disallow"):
            value, coil_gradient, inner_gradient = value_and_gradients(
                coil_dofs,
                inner_dofs,
                True,
                True,
            )
            jax.block_until_ready((value, coil_gradient, inner_gradient))

    np.testing.assert_allclose(np.asarray(value), -3.25)
    np.testing.assert_array_equal(np.asarray(coil_gradient), np.ones(2))
    np.testing.assert_array_equal(np.asarray(inner_gradient), np.ones(6))


def _make_public_boozer_fixture() -> tuple[
    BoozerSurfaceJAX,
    BiotSavartJAX,
    CurveXYZFourier,
]:
    ncoils = 2
    nfp = 2
    stellsym = True
    major_radius = 1.0
    minor_radius = 0.5
    curve_order = 3
    base_curves = create_equally_spaced_curves(
        ncoils,
        nfp,
        stellsym=stellsym,
        R0=major_radius,
        R1=minor_radius,
        order=curve_order,
    )
    base_currents = [Current(1.0e5) for _ in range(ncoils)]
    for current in base_currents:
        current.fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)

    mpol = 2
    ntor = 2
    nphi = 2 * ntor + 1
    ntheta = 2 * mpol + 1
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=stellsym,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, ntheta, endpoint=False),
    )
    reference_surface = SurfaceRZFourier(
        nfp=nfp,
        stellsym=stellsym,
        mpol=1,
        ntor=0,
        quadpoints_phi=surface.quadpoints_phi,
        quadpoints_theta=surface.quadpoints_theta,
    )
    reference_surface.set_rc(0, 0, major_radius)
    reference_surface.set_rc(1, 0, 0.15)
    reference_surface.set_zs(1, 0, 0.15)
    surface.least_squares_fit(reference_surface.gamma())

    biotsavart = BiotSavartJAX(coils)
    volume = Volume(surface)
    boozer_surface = BoozerSurfaceJAX(
        biotsavart,
        surface,
        volume,
        volume.J(),
        constraint_weight=1.0,
        options={
            "verbose": False,
            "bfgs_maxiter": 300,
            "bfgs_tol": 1.0e-8,
            "newton_maxiter": 20,
            "newton_tol": 1.0e-9,
            "optimizer_backend": "ondevice",
            "weight_inv_modB": True,
        },
    )
    permeability = 4.0 * np.pi * 1.0e-7
    initial_G = permeability * sum(
        abs(coil.current.get_value()) for coil in coils
    )
    result = boozer_surface.run_code(0.3, initial_G)
    assert result is not None and result.get("success", False)
    return boozer_surface, biotsavart, base_curves[0]


def _make_public_boozer_residual() -> BoozerResidualJAX:
    boozer_surface, biotsavart, _base_curve = _make_public_boozer_fixture()
    return BoozerResidualJAX(boozer_surface, biotsavart)


def test_public_boozer_residual_fused_gradient_obeys_strict_gpu_transfer_guard(
    monkeypatch,
    request,
) -> None:
    enable_strict_parity_backend(monkeypatch, request, "gpu")
    monkeypatch.setattr(
        surface_objectives_module,
        "_solve_boozer_adjoint",
        lambda _adjoint_state, rhs: rhs - rhs,
    )
    monkeypatch.setattr(
        surface_objectives_module,
        "_adjoint_coil_dofs_gradient",
        lambda _stream_group_vjps, _adjoint, _biotsavart, coil_dofs: (
            coil_dofs - coil_dofs
        ),
    )

    with parity_default_device("gpu"):
        objective = _make_public_boozer_residual()
        with jax.transfer_guard("disallow"):
            gradient = objective.dJ_by_dcoil_dofs()
            jax.block_until_ready(gradient)

    assert np.isfinite(objective.J())
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_public_fused_objectives_rebuild_after_coil_dof_fix_and_unfix(
    monkeypatch,
    request,
) -> None:
    enable_strict_parity_backend(monkeypatch, request, "gpu")
    monkeypatch.setattr(
        surface_objectives_module,
        "_solve_boozer_adjoint",
        lambda _adjoint_state, rhs: rhs - rhs,
    )
    monkeypatch.setattr(
        surface_objectives_module,
        "_adjoint_coil_dofs_gradient",
        lambda _stream_group_vjps, _adjoint, _biotsavart, coil_dofs: (
            coil_dofs - coil_dofs
        ),
    )

    with parity_default_device("gpu"):
        boozer_surface, biotsavart, base_curve = _make_public_boozer_fixture()
        solved_state = boozer_surface.get_solved_runtime_state()
        adjoint_state = boozer_surface.get_adjoint_runtime_state()
        monkeypatch.setattr(
            surface_objectives_module,
            "_resolved_boozer_solved_runtime_state",
            lambda _boozer_surface: solved_state,
        )
        monkeypatch.setattr(
            surface_objectives_module,
            "_resolved_boozer_adjoint_runtime_state",
            lambda _boozer_surface: adjoint_state,
        )
        objectives = (
            BoozerResidualJAX(boozer_surface, biotsavart),
            NonQuasiSymmetricRatioJAX(
                boozer_surface,
                biotsavart,
                sDIM=2,
            ),
        )
        base_curve_lineage_index = next(
            index
            for index, optimizable in enumerate(biotsavart.unique_dof_lineage)
            if optimizable is base_curve
        )
        lineage_before_base_curve = biotsavart.unique_dof_lineage[
            :base_curve_lineage_index
        ]
        base_free_start = sum(
            opt.local_dof_size for opt in lineage_before_base_curve
        )
        base_full_start = sum(
            opt.local_full_dof_size for opt in lineage_before_base_curve
        )

        with jax.transfer_guard("disallow"):
            initial_gradients = tuple(
                objective.dJ_by_dcoil_dofs() for objective in objectives
            )
            jax.block_until_ready(initial_gradients)

        base_curve.fix(0)
        with jax.transfer_guard("disallow"):
            fixed_gradients = tuple(
                objective.dJ_by_dcoil_dofs() for objective in objectives
            )
            jax.block_until_ready(fixed_gradients)

        fixed_value = float(base_curve.local_full_x[0])
        extraction_spec_before_value_change = biotsavart.coil_dof_extraction_spec()
        full_coil_dofs = np.asarray(biotsavart.full_x, dtype=np.float64).copy()
        full_coil_dofs[base_full_start] = fixed_value + 1.0e-3
        biotsavart.full_x = full_coil_dofs
        assert (
            biotsavart.coil_dof_extraction_spec()
            is not extraction_spec_before_value_change
        )
        fresh_objectives = (
            BoozerResidualJAX(boozer_surface, biotsavart),
            NonQuasiSymmetricRatioJAX(
                boozer_surface,
                biotsavart,
                sDIM=2,
            ),
        )
        with jax.transfer_guard("disallow"):
            changed_gradients = tuple(
                objective.dJ_by_dcoil_dofs() for objective in objectives
            )
            fresh_gradients = tuple(
                objective.dJ_by_dcoil_dofs() for objective in fresh_objectives
            )
            jax.block_until_ready((changed_gradients, fresh_gradients))

        full_coil_dofs[base_full_start] = fixed_value
        biotsavart.full_x = full_coil_dofs
        base_curve.unfix(0)
        with jax.transfer_guard("disallow"):
            restored_gradients = tuple(
                objective.dJ_by_dcoil_dofs() for objective in objectives
            )
            jax.block_until_ready(restored_gradients)

    for initial, fixed, restored in zip(
        initial_gradients,
        fixed_gradients,
        restored_gradients,
    ):
        assert fixed.shape == (initial.size - 1,)
        assert restored.shape == initial.shape
        assert np.all(np.isfinite(np.asarray(fixed)))
        assert np.all(np.isfinite(np.asarray(restored)))
        np.testing.assert_allclose(
            np.asarray(fixed),
            np.delete(np.asarray(initial), base_free_start),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            np.asarray(restored),
            np.asarray(initial),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
    for changed, fresh in zip(changed_gradients, fresh_gradients):
        np.testing.assert_allclose(
            np.asarray(changed),
            np.asarray(fresh),
            rtol=1.0e-10,
            atol=1.0e-10,
        )
