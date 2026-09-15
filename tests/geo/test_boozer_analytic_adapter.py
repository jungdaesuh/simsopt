"""Native label, pin, and field-chain oracles for analytic Boozer operators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import enable_non_strict_jax_backend, parity_device
from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.geo import (
    Area,
    BoozerSurface,
    SurfaceXYZTensorFourier,
    ToroidalFlux,
    Volume,
)
from simsopt_jax.geo.optimizers.native_ls_newton import newton_ls_native_dense
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.nested_ls_ncsx import (
    NcsxInnerReport,
    ncsx_banana_run_code,
)


@pytest.fixture(params=("cpu", "gpu"), autouse=True)
def analytic_backend(monkeypatch, request):
    device = parity_device(request.param)
    enable_non_strict_jax_backend(monkeypatch, request, f"jax_{request.param}_parity")
    with jax.default_device(device):
        yield device


def _problem(label_kind, stellsym, *, exact=False, native_scale=False):
    configuration = (
        {}
        if native_scale
        else {
            "coil_order": 3,
            "magnetic_axis_order": 3,
            "points_per_period": 8,
        }
    )
    _, currents, axis, nfp, field = get_data("ncsx", **configuration)
    resolution = 6 if native_scale else 1
    nquad = 2 * resolution + 1
    surface_args = {
        "mpol": resolution,
        "ntor": resolution,
        "stellsym": stellsym,
        "nfp": nfp,
        "quadpoints_phi": np.linspace(0, 1 / nfp, nquad, endpoint=False),
        "quadpoints_theta": np.linspace(0, 1, nquad, endpoint=False),
    }
    native_surface = SurfaceXYZTensorFourier(**surface_args)
    native_surface.fit_to_curve(axis, 0.1, flip_theta=True)
    device_surface = SurfaceXYZTensorFourier(**surface_args)
    device_surface.set_dofs(native_surface.get_dofs())
    if label_kind == "volume":
        native_label, device_label = Volume(native_surface), Volume(device_surface)
    elif label_kind == "area":
        native_label, device_label = Area(native_surface), Area(device_surface)
    else:
        native_label = ToroidalFlux(native_surface, BiotSavart(field.coils))
        device_label = ToroidalFlux(device_surface, BiotSavart(field.coils))
    target = native_label.J() * 1.01
    constraint_weight = None if exact else 11.1232
    native = BoozerSurface(
        field,
        native_surface,
        native_label,
        target,
        constraint_weight=constraint_weight,
    )
    device = BoozerSurfaceJAX(
        BiotSavartJAX(field.coils),
        device_surface,
        device_label,
        target,
        constraint_weight=constraint_weight,
    )
    G = 4e-7 * np.pi * nfp * sum(abs(current.get_value()) for current in currents)
    return native, device, G


@pytest.mark.parametrize(
    "label_kind,stellsym,optimize_G,weighted",
    [
        ("volume", True, True, False),
        ("area", False, False, True),
        ("toroidal_flux", True, True, True),
    ],
)
def test_analytic_ls_full_native_operator(
    monkeypatch,
    request,
    label_kind,
    stellsym,
    optimize_G,
    weighted,
):
    native, device, G = _problem(label_kind, stellsym)
    tail = [-0.37, G] if optimize_G else [-0.37]
    x = np.concatenate((native.surface.get_dofs(), tail))
    reference = native.boozer_penalty_constraints_vectorized(
        x,
        derivatives=2,
        constraint_weight=11.1232,
        optimize_G=optimize_G,
        weight_inv_modB=weighted,
    )
    value_grad, value_grad_hessian = device._make_analytic_penalty_derivatives(
        optimize_G,
        weighted,
    )
    actual = jax.jit(value_grad_hessian)(jnp.asarray(x), device.coil_set_spec)
    first = jax.jit(value_grad)(jnp.asarray(x), device.coil_set_spec)
    for observed, expected in zip(actual, reference):
        np.testing.assert_allclose(observed, expected, rtol=2e-10, atol=2e-11)
    for observed, expected in zip(first, actual[:2]):
        np.testing.assert_allclose(observed, expected, rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize("stellsym,weighted", [(True, False), (False, True)])
def test_analytic_exact_masks_label_and_axis(monkeypatch, request, stellsym, weighted):
    native, device, G = _problem("volume", stellsym, exact=True)
    device.options["weight_inv_modB"] = weighted
    x = jnp.asarray(np.concatenate((native.surface.get_dofs(), [-0.37, G])))
    reference_fn = device._make_exact_residual(device._compute_stellsym_mask_indices())
    reference = (reference_fn(x), jax.jacfwd(reference_fn)(x))
    analytic = device._make_analytic_exact_value_jacobian(weighted)
    actual = jax.jit(analytic)(x, device.coil_set_spec)
    for observed, expected in zip(actual, reference):
        np.testing.assert_allclose(observed, expected, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("native_scale", (False, True))
def test_analytic_c2_inner_solve_matches_native(monkeypatch, request, native_scale):
    native, device, G = _problem("volume", True, exact=True, native_scale=native_scale)
    initial_dofs = np.array(native.surface.get_dofs(), copy=True)
    device.options["weight_inv_modB"] = False
    device.options["newton_maxiter"] = 20
    device.options["newton_tol"] = 1e-13
    reference = native.solve_residual_equation_exactly_newton(
        iota=-0.406,
        G=G,
        maxiter=20,
        tol=1e-13,
    )
    route = device._make_run_code_traceable_exact_benchmark_variant("C2", analytic=True)
    solve_inputs = (
        device.coil_set_spec,
        None,
        jnp.asarray(initial_dofs),
        jnp.asarray(-0.406),
        jnp.asarray(G),
    )
    with jax.transfer_guard("disallow"):
        actual = route.compiled_kernel(*solve_inputs)
        jax.block_until_ready(actual)
    assert reference["success"]
    assert bool(actual["success"])
    expected_x = np.concatenate(
        (native.surface.get_dofs(), [reference["iota"], reference["G"]])
    )
    np.testing.assert_allclose(actual["x"], expected_x, rtol=1e-10, atol=1e-11)
    final_residual, final_jacobian = device._make_analytic_exact_value_jacobian(False)(
        actual["x"],
        device.coil_set_spec,
    )
    assert float(jnp.linalg.norm(final_residual)) <= 1e-13
    np.testing.assert_allclose(
        actual["jacobian"], final_jacobian, rtol=2e-12, atol=2e-13
    )


@pytest.mark.parametrize("weighted,expected_success", ((False, False), (True, True)))
def test_analytic_ls_newton_matches_native_from_common_seed(weighted, expected_success):
    native, device, G = _problem("volume", True)
    pre_solve = native.minimize_boozer_penalty_constraints_LBFGS(
        iota=-0.406,
        G=G,
        constraint_weight=11.1232,
        tol=1e-8,
        maxiter=1500,
        limited_memory=False,
        weight_inv_modB=weighted,
    )
    seed = np.concatenate(
        (native.surface.get_dofs(), [pre_solve["iota"], pre_solve["G"]])
    )
    native.need_to_run_code = True
    reference = native.minimize_boozer_penalty_constraints_newton(
        iota=seed[-2],
        G=seed[-1],
        constraint_weight=11.1232,
        tol=1e-11,
        maxiter=40,
        weight_inv_modB=weighted,
    )
    _, derivatives = device._make_analytic_penalty_derivatives(True, weighted)

    @jax.jit
    def solve(initial_x, coil_spec):
        return newton_ls_native_dense(
            derivatives, initial_x, maxiter=40, tol=1e-11, args=(coil_spec,)
        )

    device_seed = jnp.asarray(seed)
    with jax.transfer_guard("disallow"):
        actual = solve(device_seed, device.coil_set_spec)
        jax.block_until_ready(actual)
    # This coarse unweighted fixture is a native cap-exhaustion case; only the
    # weighted case establishes a successful solve at the unchanged tolerance.
    assert bool(reference["success"]) is expected_success
    assert bool(actual["success"]) is expected_success
    expected_x = np.concatenate(
        (native.surface.get_dofs(), [reference["iota"], reference["G"]])
    )
    np.testing.assert_allclose(actual["x"], expected_x, rtol=2e-10, atol=2e-11)
    if expected_success:
        assert float(jnp.linalg.norm(actual["grad"])) <= 1e-11
    else:
        assert reference["iter"] == int(actual["nit"]) < 40
    np.testing.assert_allclose(
        actual["hessian"], reference["hessian"], rtol=2e-10, atol=2e-11
    )


@pytest.mark.parametrize("weighted", (True,))
def test_analytic_newton_method_matches_native_from_common_seed(weighted):
    native, device, G = _problem("volume", True)
    pre_solve = native.minimize_boozer_penalty_constraints_LBFGS(
        iota=-0.406,
        G=G,
        constraint_weight=11.1232,
        tol=1e-8,
        maxiter=1500,
        limited_memory=False,
        weight_inv_modB=weighted,
    )
    seed_dofs = np.array(native.surface.get_dofs(), copy=True)
    native.need_to_run_code = True
    reference = native.minimize_boozer_penalty_constraints_newton(
        iota=pre_solve["iota"],
        G=pre_solve["G"],
        constraint_weight=11.1232,
        tol=1e-11,
        maxiter=40,
        weight_inv_modB=weighted,
    )
    device._set_surface_dofs(jnp.asarray(seed_dofs))
    device.need_to_run_code = True
    actual = device._minimize_boozer_penalty_constraints_newton_analytic(
        constraint_weight=11.1232,
        iota=pre_solve["iota"],
        G=pre_solve["G"],
        tol=1e-11,
        maxiter=40,
        verbose=False,
        weight_inv_modB=weighted,
    )
    assert bool(reference["success"]) and bool(actual["success"])
    assert int(actual["iter"]) == int(reference["iter"])
    np.testing.assert_allclose(
        actual["iota"], reference["iota"], rtol=1e-10, atol=1e-11
    )
    np.testing.assert_allclose(actual["G"], reference["G"], rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(
        np.asarray(device.surface.get_dofs()),
        np.asarray(native.surface.get_dofs()),
        rtol=1e-10,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        np.asarray(actual["hessian"]), reference["hessian"], rtol=2e-10, atol=2e-11
    )
    assert actual["PLU"] is not None
    assert actual["vjp"] is not None
    assert actual["type"] == "ls"
    assert device.need_to_run_code is False
    assert device.res is actual


def test_ncsx_banana_analytic_matches_native_run_code_from_common_start():
    native, device, G = _problem("volume", True)
    for boozer in (native, device):
        boozer.options["verbose"] = False
        boozer.options["bfgs_maxiter"] = 20
    reference = native.run_code(-0.406, G)
    report = NcsxInnerReport()
    actual = ncsx_banana_run_code(
        device, -0.406, G, derivative_assembly="analytic", report=report
    )
    assert bool(reference["success"]) and bool(actual["success"])
    assert report.bfgs_nit >= 1
    assert report.bfgs_seconds > 0.0 and report.newton_seconds > 0.0
    np.testing.assert_allclose(actual["iota"], reference["iota"], rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(actual["G"], reference["G"], rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(
        np.asarray(device.surface.get_dofs()),
        np.asarray(native.surface.get_dofs()),
        rtol=1e-8,
        atol=1e-9,
    )
    with pytest.raises(ValueError, match="derivative_assembly"):
        ncsx_banana_run_code(device, -0.406, G, derivative_assembly="symbolic")
