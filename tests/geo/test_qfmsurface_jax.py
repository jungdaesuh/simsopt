"""QFM JAX solver orchestration tests."""

from __future__ import annotations

from jax_test_support import (
    fixture_jax_runtime_guard,  # noqa: F401
    host_array,
    host_scalar,
)

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsopt_jax.geo.optimizers.private._bfgs as _private_bfgs

from simsopt.configs.zoo import get_data
from simsopt.field.biotsavart import BiotSavart
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt.geo.qfmsurface import QfmSurface
from simsopt.geo.surfaceobjectives import Area, Volume
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt_jax.core import qfm_solver as qfm_solver_module
from simsopt_jax.core.qfm_solver import (
    qfm_augmented_lagrangian_solve_jax,
    qfm_exact_kkt_residual_jax_from_dofs,
    qfm_penalty_jax_from_dofs,
    qfm_penalty_solve_jax,
    qfm_penalty_value_and_grad_jax_from_dofs,
    qfm_residual_jax_from_dofs,
)
from simsopt_jax.core.specs import (
    make_surface_xyz_fourier_spec,
    make_surface_xyz_tensor_fourier_spec,
)
from simsopt_jax.core.surface_rzfourier import surface_rz_fourier_spec_from_dofs

from .surface_test_helpers import get_surface


def _make_ncsx_rz_qfm_surface():
    _base_curves, _base_currents, magnetic_axis, nfp, biotsavart = get_data("ncsx")
    phis = np.linspace(0.0, 1.0 / nfp, 6, endpoint=False)
    thetas = np.linspace(0.0, 1.0, 6, endpoint=False)
    surface = SurfaceRZFourier(
        mpol=1,
        ntor=1,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    surface.fit_to_curve(magnetic_axis, 0.2, flip_theta=True)
    return biotsavart, surface


def _make_qfm_case():
    biotsavart, surface = _make_ncsx_rz_qfm_surface()
    return BiotSavartJAX(biotsavart.coils), surface


def _make_label_grid_qfm_cpu_case():
    _base_curves, _base_currents, _magnetic_axis, nfp, biotsavart = get_data("ncsx")
    phis = np.linspace(0.0, 1.0 / nfp, 6, endpoint=False)
    thetas = np.linspace(0.0, 1.0, 5, endpoint=False)
    surface = SurfaceRZFourier(
        mpol=3,
        ntor=2,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    dofs = np.asarray(surface.get_dofs(), dtype=np.float64)
    surface.x = dofs + 0.02 * np.sin(np.arange(dofs.size, dtype=np.float64))
    return biotsavart, surface


def _make_label_grid_qfm_case():
    biotsavart, surface = _make_label_grid_qfm_cpu_case()
    return BiotSavartJAX(biotsavart.coils), surface


def _coil_set_spec(biotsavart):
    return biotsavart.coil_set_spec_from_dofs(
        jnp.asarray(biotsavart.x, dtype=jnp.float64)
    )


def _surface_spec(surface):
    surface_type = type(surface).__name__
    if surface_type == "SurfaceRZFourier":
        return surface_rz_fourier_spec_from_dofs(
            surface.get_dofs(),
            quadpoints_phi=surface.quadpoints_phi,
            quadpoints_theta=surface.quadpoints_theta,
            mpol=surface.mpol,
            ntor=surface.ntor,
            nfp=surface.nfp,
            stellsym=surface.stellsym,
        )
    if surface_type == "SurfaceXYZFourier":
        return make_surface_xyz_fourier_spec(
            dofs=surface.get_dofs(),
            quadpoints_phi=surface.quadpoints_phi,
            quadpoints_theta=surface.quadpoints_theta,
            nfp=surface.nfp,
            stellsym=surface.stellsym,
            mpol=surface.mpol,
            ntor=surface.ntor,
        )
    if surface_type == "SurfaceXYZTensorFourier":
        return make_surface_xyz_tensor_fourier_spec(
            dofs=surface.get_dofs(),
            quadpoints_phi=surface.quadpoints_phi,
            quadpoints_theta=surface.quadpoints_theta,
            nfp=surface.nfp,
            stellsym=surface.stellsym,
            mpol=surface.mpol,
            ntor=surface.ntor,
            clamped_dims=tuple(getattr(surface, "clamped_dims", (False, False, False))),
        )
    raise TypeError(f"Unsupported surface type for explicit spec: {surface_type}")


def _make_qfm_inputs():
    biotsavart, surface = _make_qfm_case()
    dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)
    return biotsavart, surface, dofs, _coil_set_spec(biotsavart)


def test_qfm_limited_memory_optimizer_remains_fail_closed_without_crossover() -> None:
    """R16: do not expose L-BFGS before a qualifying device-resident crossover."""
    with pytest.raises(NotImplementedError, match="support optimizer='bfgs' only"):
        qfm_solver_module._require_bfgs_optimizer("lbfgs")


def _make_test_qfm_xyz_volume_case():
    _base_curves, _base_currents, magnetic_axis, nfp, biotsavart = get_data("ncsx")
    phis = np.linspace(0.0, 1.0 / nfp, 20, endpoint=False)
    thetas = np.linspace(0.0, 1.0, 20, endpoint=False)
    surface = get_surface(
        "SurfaceXYZFourier",
        True,
        phis=phis,
        thetas=thetas,
        ntor=4,
        mpol=4,
    )
    surface.fit_to_curve(magnetic_axis, 0.2)
    return biotsavart, surface


def test_qfm_bfgs_curvature_floor_rejects_float32_boundary() -> None:
    s = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    y = jnp.asarray([1.0e-5, 1.0], dtype=jnp.float32)

    qfm_valid = qfm_solver_module._bfgs_has_valid_curvature(s, y)
    # Private helper returns (rho_inv, rho, valid); never a 4-tuple.
    _, _, private_valid = _private_bfgs._bfgs_curvature_terms(
        s,
        y,
        x_dtype=s.dtype,
    )

    assert bool(qfm_valid) is False
    assert bool(private_valid) is False


def _cpu_label_value(label, dofs: object) -> float:
    label.surface.x = host_array(dofs)
    return label.J()


def test_qfm_penalty_solve_jax_reduces_fixed_state_penalty() -> None:
    """Oracle: the same pure QFM penalty kernel before and after the solve."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = 0.98 * Area(surface).J()
    initial = qfm_penalty_jax_from_dofs(
        _surface_spec(surface),
        dofs,
        coil_set_spec,
        label="area",
        label_spec=_surface_spec(surface),
        label_coil_set_spec=coil_set_spec,
        targetlabel=target,
        constraint_weight=1.0,
    )

    final_dofs, info = qfm_penalty_solve_jax(
        _surface_spec(surface),
        coil_set_spec,
        "area",
        target,
        1.0,
        dofs,
        label_spec=_surface_spec(surface),
        label_coil_set_spec=coil_set_spec,
        max_iter=5,
        tol=1e-8,
    )

    assert final_dofs.shape == dofs.shape
    assert host_scalar(info.penalty_value) < host_scalar(initial)
    assert host_array(info.gradient).shape == tuple(dofs.shape)


def test_qfm_penalty_solve_jax_not_worse_than_host_lbfgsb_diagnostic() -> None:
    """Diagnostic: JAX penalty solve is not worse than host LBFGS-B residual."""
    biotsavart_cpu, surface_cpu = _make_test_qfm_xyz_volume_case()
    _biotsavart_src, surface_jax = _make_test_qfm_xyz_volume_case()
    label_cpu = Volume(surface_cpu)
    target = label_cpu.J()
    qfm_cpu = QfmSurface(
        BiotSavart(biotsavart_cpu.coils),
        surface_cpu,
        label_cpu,
        target,
    )

    cpu_result = qfm_cpu.minimize_qfm_penalty_constraints_LBFGS(
        tol=1e-8,
        maxiter=200,
        constraint_weight=1.0,
    )
    cpu_qfm_residual = qfm_cpu.qfm_objective(surface_cpu.get_dofs())
    biotsavart_jax = BiotSavartJAX(biotsavart_cpu.coils)
    coil_set_spec = _coil_set_spec(biotsavart_jax)
    final_dofs, info = qfm_penalty_solve_jax(
        _surface_spec(surface_jax),
        coil_set_spec,
        "volume",
        target,
        1.0,
        jnp.asarray(surface_jax.get_dofs(), dtype=jnp.float64),
        label_spec=_surface_spec(surface_jax),
        label_coil_set_spec=coil_set_spec,
        max_iter=400,
        tol=1e-8,
    )

    assert cpu_result["success"]
    assert final_dofs.shape == tuple(surface_jax.get_dofs().shape)
    assert host_scalar(info.qfm_value) <= cpu_qfm_residual * (1.0 + 1.0e-6)


def test_qfm_augmented_lagrangian_meets_upstream_exact_acceptance() -> None:
    """AL exact path meets upstream QFM acceptance after the host LBFGS warm start."""
    biotsavart_cpu, warm_surface = _make_test_qfm_xyz_volume_case()
    label = Volume(warm_surface)
    target = label.J()
    warm_qfm = QfmSurface(
        BiotSavart(biotsavart_cpu.coils),
        warm_surface,
        label,
        target,
    )
    warm_result = warm_qfm.minimize_qfm_penalty_constraints_LBFGS(
        tol=1e-8,
        maxiter=1000,
        constraint_weight=1.0,
    )
    warm_dofs = np.asarray(warm_surface.get_dofs(), dtype=np.float64)

    _biotsavart_src, host_surface = _make_test_qfm_xyz_volume_case()
    host_surface.x = warm_dofs.copy()
    host_qfm = QfmSurface(
        BiotSavart(biotsavart_cpu.coils),
        host_surface,
        Volume(host_surface),
        target,
    )
    host_result = host_qfm.minimize_qfm_exact_constraints_SLSQP(tol=1e-9, maxiter=1000)
    host_qfm_residual = host_qfm.qfm_objective(host_surface.get_dofs())
    host_label_residual = Volume(host_surface).J() - target

    _biotsavart_src, jax_surface = _make_test_qfm_xyz_volume_case()
    jax_surface.x = warm_dofs.copy()
    biotsavart_jax = BiotSavartJAX(biotsavart_cpu.coils)
    coil_set_spec = _coil_set_spec(biotsavart_jax)
    al_dofs, al_info = qfm_augmented_lagrangian_solve_jax(
        _surface_spec(jax_surface),
        coil_set_spec,
        "volume",
        target,
        jnp.asarray(jax_surface.get_dofs(), dtype=jnp.float64),
        label_spec=_surface_spec(jax_surface),
        label_coil_set_spec=coil_set_spec,
        max_outer=3,
        inner_max_iter=200,
        tol=1e-8,
    )

    assert warm_result["success"]
    assert host_result["success"]
    assert host_qfm_residual < 1.0e-5
    assert abs(host_label_residual) < 3.0e-5
    assert abs(host_scalar(al_info.label_residual)) <= 1.0e-6
    assert host_scalar(al_info.qfm_value) < 1.0e-5
    assert host_scalar(al_info.qfm_value) <= host_qfm_residual
    assert al_dofs.shape == tuple(jax_surface.get_dofs().shape)


def test_qfm_augmented_lagrangian_kkt_diagnostic_no_worse_than_host_slsqp() -> None:
    """Diagnostic: natural equality KKT residual, not host SLSQP DOF identity."""
    biotsavart_cpu, warm_surface = _make_test_qfm_xyz_volume_case()
    label = Volume(warm_surface)
    target = label.J()
    warm_qfm = QfmSurface(
        BiotSavart(biotsavart_cpu.coils),
        warm_surface,
        label,
        target,
    )
    warm_result = warm_qfm.minimize_qfm_penalty_constraints_LBFGS(
        tol=1e-8,
        maxiter=1000,
        constraint_weight=1.0,
    )
    warm_dofs = np.asarray(warm_surface.get_dofs(), dtype=np.float64)

    _biotsavart_src, host_surface = _make_test_qfm_xyz_volume_case()
    host_surface.x = warm_dofs.copy()
    host_qfm = QfmSurface(
        BiotSavart(biotsavart_cpu.coils),
        host_surface,
        Volume(host_surface),
        target,
    )
    host_result = host_qfm.minimize_qfm_exact_constraints_SLSQP(tol=1e-9, maxiter=1000)

    _biotsavart_src, jax_surface = _make_test_qfm_xyz_volume_case()
    jax_surface.x = warm_dofs.copy()
    biotsavart_jax = BiotSavartJAX(biotsavart_cpu.coils)
    coil_set_spec = _coil_set_spec(biotsavart_jax)
    surface_spec = _surface_spec(jax_surface)
    al_dofs, al_info = qfm_augmented_lagrangian_solve_jax(
        surface_spec,
        coil_set_spec,
        "volume",
        target,
        jnp.asarray(jax_surface.get_dofs(), dtype=jnp.float64),
        label_spec=surface_spec,
        label_coil_set_spec=coil_set_spec,
        max_outer=3,
        inner_max_iter=200,
        tol=1e-8,
    )

    host_kkt = qfm_exact_kkt_residual_jax_from_dofs(
        surface_spec,
        jnp.asarray(host_surface.get_dofs(), dtype=jnp.float64),
        coil_set_spec,
        label="volume",
        label_spec=surface_spec,
        label_coil_set_spec=coil_set_spec,
        targetlabel=target,
    )
    al_kkt = qfm_exact_kkt_residual_jax_from_dofs(
        surface_spec,
        al_dofs,
        coil_set_spec,
        label="volume",
        label_spec=surface_spec,
        label_coil_set_spec=coil_set_spec,
        targetlabel=target,
    )

    assert warm_result["success"]
    assert host_result["success"]
    assert abs(host_scalar(al_info.label_residual)) <= 1.0e-6
    assert host_scalar(host_kkt.label_gradient_norm) > 1.0
    assert host_scalar(al_kkt.label_gradient_norm) > 1.0
    assert host_scalar(al_kkt.feasibility_abs) <= host_scalar(host_kkt.feasibility_abs)
    assert host_scalar(al_kkt.stationarity_inf) <= host_scalar(
        host_kkt.stationarity_inf
    ) * (1.0 + 1.0e-8)


def test_qfm_augmented_lagrangian_success_uses_absolute_kkt() -> None:
    """KKT-passing AL results are successful at the public solver surface."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = 0.99 * Area(surface).J()
    surface_spec = _surface_spec(surface)
    perturbation = 1.0e-3 * jnp.sin(jnp.arange(dofs.size, dtype=jnp.float64))
    final_dofs, info = qfm_augmented_lagrangian_solve_jax(
        surface_spec,
        coil_set_spec,
        "area",
        target,
        dofs + perturbation,
        label_spec=surface_spec,
        label_coil_set_spec=coil_set_spec,
        max_outer=5,
        inner_max_iter=100,
        tol=1.0e-6,
    )
    kkt = qfm_exact_kkt_residual_jax_from_dofs(
        surface_spec,
        final_dofs,
        coil_set_spec,
        label="area",
        label_spec=surface_spec,
        label_coil_set_spec=coil_set_spec,
        targetlabel=target,
    )

    assert bool(host_scalar(info.success))
    assert host_scalar(kkt.feasibility_abs) <= 1.0e-6
    assert host_scalar(kkt.stationarity_inf) <= 1.0e-6


def test_qfm_augmented_lagrangian_rejects_feasible_nonstationary_state() -> None:
    """Label feasibility alone is not enough for exact-path AL success."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = 0.99 * Area(surface).J()
    surface_spec = _surface_spec(surface)
    final_dofs, info = qfm_augmented_lagrangian_solve_jax(
        surface_spec,
        coil_set_spec,
        "area",
        target,
        dofs,
        label_spec=surface_spec,
        label_coil_set_spec=coil_set_spec,
        max_outer=2,
        inner_max_iter=20,
        tol=2.0e-4,
    )
    kkt = qfm_exact_kkt_residual_jax_from_dofs(
        surface_spec,
        final_dofs,
        coil_set_spec,
        label="area",
        label_spec=surface_spec,
        label_coil_set_spec=coil_set_spec,
        targetlabel=target,
    )

    assert host_scalar(kkt.feasibility_abs) <= 2.0e-4
    assert host_scalar(kkt.stationarity_inf) > 2.0e-4
    assert not bool(host_scalar(info.success))


def test_qfm_augmented_lagrangian_branch_stability_uses_kkt_invariants() -> None:
    """Small warm-start perturbations preserve objective, label, and KKT invariants."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = 0.99 * Area(surface).J()
    surface_spec = _surface_spec(surface)
    perturbation = jnp.sin(jnp.arange(dofs.size, dtype=jnp.float64))
    results = []
    for scale in (0.0, 1.0e-4, 1.0e-3):
        final_dofs, info = qfm_augmented_lagrangian_solve_jax(
            surface_spec,
            coil_set_spec,
            "area",
            target,
            dofs + scale * perturbation,
            label_spec=surface_spec,
            label_coil_set_spec=coil_set_spec,
            max_outer=5,
            inner_max_iter=100,
            tol=1.0e-6,
        )
        kkt = qfm_exact_kkt_residual_jax_from_dofs(
            surface_spec,
            final_dofs,
            coil_set_spec,
            label="area",
            label_spec=surface_spec,
            label_coil_set_spec=coil_set_spec,
            targetlabel=target,
        )
        assert bool(host_scalar(info.success))
        assert host_scalar(kkt.feasibility_abs) <= 1.0e-6
        assert host_scalar(kkt.stationarity_inf) <= 1.0e-6
        results.append(
            (
                host_scalar(info.qfm_value),
                host_scalar(info.label_residual),
                host_scalar(kkt.stationarity_inf),
            )
        )

    qfm_values, label_residuals, stationarity_values = zip(*results, strict=True)
    np.testing.assert_allclose(qfm_values, qfm_values[0], rtol=1.0e-7, atol=1.0e-12)
    np.testing.assert_allclose(
        label_residuals,
        label_residuals[0],
        rtol=0.0,
        atol=1.0e-9,
    )
    assert np.ptp(np.asarray(stationarity_values)) <= 5.0e-7


def test_qfm_penalty_fixed_state_gradient_matches_centered_fd() -> None:
    """Derivative-heavy lane: fixed-state JAX gradient matches FD."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = 0.98 * Area(surface).J()
    surface_spec = _surface_spec(surface)
    value, gradient = qfm_penalty_value_and_grad_jax_from_dofs(
        surface_spec,
        dofs,
        coil_set_spec,
        label="area",
        label_spec=surface_spec,
        label_coil_set_spec=coil_set_spec,
        targetlabel=target,
        constraint_weight=1.0,
    )
    # h=2^-18 is still in O(h^2) FD truncation (rel 1.19e-8 > rtol 1e-8 on
    # dof 6). h=2^-20 is inside the same rtol/atol without loosening.
    step = 2.0**-20
    finite_difference_gradient = []
    for idx in range(dofs.size):
        basis = np.zeros(dofs.size, dtype=np.float64)
        basis[idx] = 1.0
        basis_jax = jnp.asarray(basis, dtype=jnp.float64)
        value_plus = qfm_penalty_jax_from_dofs(
            surface_spec,
            dofs + step * basis_jax,
            coil_set_spec,
            label="area",
            label_spec=surface_spec,
            label_coil_set_spec=coil_set_spec,
            targetlabel=target,
            constraint_weight=1.0,
        )
        value_minus = qfm_penalty_jax_from_dofs(
            surface_spec,
            dofs - step * basis_jax,
            coil_set_spec,
            label="area",
            label_spec=surface_spec,
            label_coil_set_spec=coil_set_spec,
            targetlabel=target,
            constraint_weight=1.0,
        )
        finite_difference_gradient.append(
            (host_scalar(value_plus) - host_scalar(value_minus)) / (2.0 * step)
        )

    assert np.isfinite(host_scalar(value))
    np.testing.assert_allclose(
        np.asarray(finite_difference_gradient),
        host_array(gradient),
        rtol=1.0e-8,
        atol=1.0e-10,
    )


def test_qfm_penalty_solve_jax_transfer_guard_clean() -> None:
    """The BFGS solver core does not enter JAX's host-staging optimizer path."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = jnp.asarray(0.98 * Area(surface).J(), dtype=dofs.dtype)
    constraint_weight = jnp.asarray(1.0, dtype=dofs.dtype)

    with jax.transfer_guard("disallow"):
        final_dofs, info = qfm_penalty_solve_jax(
            _surface_spec(surface),
            coil_set_spec,
            "area",
            target,
            constraint_weight,
            dofs,
            label_spec=_surface_spec(surface),
            label_coil_set_spec=coil_set_spec,
            max_iter=1,
            tol=1e-8,
        )

    assert final_dofs.shape == dofs.shape
    assert host_array(info.gradient).shape == tuple(dofs.shape)


def test_qfm_augmented_lagrangian_solve_jax_transfer_guard_clean() -> None:
    """The AL wrapper keeps scalar updates and inner BFGS staging on device."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = jnp.asarray(0.99 * Area(surface).J(), dtype=dofs.dtype)

    with jax.transfer_guard("disallow"):
        final_dofs, info = qfm_augmented_lagrangian_solve_jax(
            _surface_spec(surface),
            coil_set_spec,
            "area",
            target,
            dofs,
            label_spec=_surface_spec(surface),
            label_coil_set_spec=coil_set_spec,
            max_outer=2,
            inner_max_iter=1,
            tol=1e-8,
        )

    assert final_dofs.shape == dofs.shape
    assert host_array(info.gradient).shape == tuple(dofs.shape)
    np.testing.assert_allclose(host_scalar(info.penalty_weight), 100.0)
    assert host_scalar(info.multiplier) != 0.0


@pytest.mark.parametrize("committed_coils", [False, True])
def test_qfm_augmented_lagrangian_compiles_its_bfgs_runner_once(
    monkeypatch, committed_coils
) -> None:
    """Every outer iteration reuses the BFGS runner's first compiled executable.

    ``jit`` keys committed and uncommitted arguments separately, so the first
    call must receive arguments committed exactly like the BFGS results and
    multiplier updates of the later iterations, or iteration two recompiles:
    with committed coils and uncommitted initial dofs, the BFGS result is
    committed while the first input was not.
    """
    runners = []
    make_runner = qfm_solver_module._make_bfgs_runner

    def recording_make_runner(*args, **kwargs):
        runner = make_runner(*args, **kwargs)
        runners.append(runner)
        return runner

    monkeypatch.setattr(qfm_solver_module, "_make_bfgs_runner", recording_make_runner)
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    if committed_coils:
        coil_set_spec = jax.device_put(coil_set_spec, jax.local_devices()[0])
    target = jnp.asarray(0.99 * Area(surface).J(), dtype=dofs.dtype)

    assert not dofs.committed
    qfm_augmented_lagrangian_solve_jax(
        _surface_spec(surface),
        coil_set_spec,
        "area",
        target,
        dofs,
        label_spec=_surface_spec(surface),
        label_coil_set_spec=coil_set_spec,
        max_outer=3,
        inner_max_iter=1,
        tol=1e-8,
    )

    (runner,) = runners
    assert runner._cache_size() == 1


def test_qfm_augmented_lagrangian_info_reports_qfm_gradient() -> None:
    """The exact-path result pairs QFM ``fun`` with the QFM objective gradient."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = Area(surface).J()

    final_dofs, info = qfm_augmented_lagrangian_solve_jax(
        _surface_spec(surface),
        coil_set_spec,
        "area",
        target,
        dofs,
        label_spec=_surface_spec(surface),
        label_coil_set_spec=coil_set_spec,
        max_outer=1,
        inner_max_iter=1,
        tol=1e-8,
    )
    expected_gradient = jax.grad(
        lambda surface_dofs: qfm_residual_jax_from_dofs(
            _surface_spec(surface),
            surface_dofs,
            coil_set_spec,
        )
    )(final_dofs)

    np.testing.assert_allclose(
        host_array(info.gradient),
        host_array(expected_gradient),
        rtol=1e-10,
        atol=1e-12,
    )


def test_qfm_augmented_lagrangian_info_keeps_augmented_value_separate() -> None:
    """AL diagnostics keep public QFM ``fun`` separate from augmented value."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = 0.99 * Area(surface).J()

    _final_dofs, info = qfm_augmented_lagrangian_solve_jax(
        _surface_spec(surface),
        coil_set_spec,
        "area",
        target,
        dofs,
        label_spec=_surface_spec(surface),
        label_coil_set_spec=coil_set_spec,
        max_outer=2,
        inner_max_iter=1,
        tol=1e-8,
    )

    np.testing.assert_allclose(host_scalar(info.fun), host_scalar(info.qfm_value))
    assert not np.isclose(
        host_scalar(info.augmented_value),
        host_scalar(info.fun),
        rtol=1e-8,
        atol=1e-12,
    )
    assert host_scalar(info.multiplier) != 0.0
    np.testing.assert_allclose(
        host_scalar(info.penalty_weight),
        100.0,
        rtol=0.0,
        atol=0.0,
    )


def test_qfm_solvers_report_label_value_on_label_surface_spec() -> None:
    """Real penalty and AL solver info use the label-owned quadrature grid."""
    biotsavart, surface = _make_label_grid_qfm_case()
    _biotsavart_cpu, label_probe_surface = _make_label_grid_qfm_cpu_case()
    label = Area(
        surface,
        nphi=4,
        ntheta=9,
        range=SurfaceRZFourier.RANGE_FULL_TORUS,
    )
    surface_spec = _surface_spec(surface)
    label_spec = _surface_spec(label.surface)
    coil_set_spec = _coil_set_spec(biotsavart)
    init_dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)
    target = label.J()
    label_probe = Area(
        label_probe_surface,
        nphi=4,
        ntheta=9,
        range=SurfaceRZFourier.RANGE_FULL_TORUS,
    )
    wrong_probe = Area(label_probe_surface)

    penalty_dofs, penalty_info = qfm_penalty_solve_jax(
        surface_spec,
        coil_set_spec,
        "area",
        target,
        1.0,
        init_dofs,
        label_spec=label_spec,
        label_coil_set_spec=coil_set_spec,
        max_iter=0,
        tol=1e-8,
    )
    al_dofs, al_info = qfm_augmented_lagrangian_solve_jax(
        surface_spec,
        coil_set_spec,
        "area",
        target,
        init_dofs,
        label_spec=label_spec,
        label_coil_set_spec=coil_set_spec,
        max_outer=1,
        inner_max_iter=1,
        tol=1e-8,
    )

    expected_penalty_label = _cpu_label_value(label_probe, penalty_dofs)
    wrong_penalty_label = _cpu_label_value(wrong_probe, penalty_dofs)
    expected_al_label = _cpu_label_value(label_probe, al_dofs)
    wrong_al_label = _cpu_label_value(wrong_probe, al_dofs)

    np.testing.assert_allclose(
        host_scalar(penalty_info.label_value),
        expected_penalty_label,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        host_scalar(al_info.label_value),
        expected_al_label,
        rtol=1e-10,
        atol=1e-12,
    )
    assert not np.isclose(
        host_scalar(penalty_info.label_value),
        wrong_penalty_label,
        rtol=1e-8,
        atol=1e-12,
    )
    assert not np.isclose(
        host_scalar(al_info.label_value),
        wrong_al_label,
        rtol=1e-8,
        atol=1e-12,
    )
