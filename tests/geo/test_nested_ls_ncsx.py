"""NCSX boozerQA nested-LS inner/outer: not F3/flat-675."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from simsopt.configs.zoo import get_data
from simsopt.geo import BoozerSurface, Volume
from simsopt_jax.parity_tolerances import parity_ladder_tolerances
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_BANANA_NEWTON_MAXITER,
    NESTED_LS_BANANA_NEWTON_TOL,
    NESTED_LS_CONSTRAINT_WEIGHT,
    NESTED_LS_WEIGHT_INV_MODB,
)
from simsopt_jax_adapters.geo import nested_ls_ncsx as ncsx_mod
from simsopt_jax_adapters.geo.nested_ls_ncsx import (
    NCSX_EVAL_TIMING_KEYS,
    NCSX_MR_WEIGHT,
    NcsxNestedLsBranchJump,
    NcsxNestedLsInnerSolveFailed,
    NcsxNestedLsSelfIntersecting,
    clone_surface_xyz_tensor_fourier,
    empty_ncsx_eval_timing,
    ncsx_native_outer_value_and_grad,
    ncsx_nested_ls_outer_value_and_grad,
    ncsx_problem_from_native_boozers,
    prepare_ncsx_nested_ls_problem,
    remap_tensor_fourier_index,
    restore_ncsx_anchor,
    run_ncsx_schur_inner,
    summarize_ncsx_eval_timings,
    upsample_surface_xyz_tensor_fourier,
)
from simsopt_jax_adapters.geo.nested_ls_reduced import (
    nested_ls_runtime_coil_closures,
    solve_projected_y,
)

from .boozersurface_jax_test_helpers import _clone_upstream_surface
from .surface_test_helpers import get_surface

_IOTA0 = -0.406


def _g_from_currents(base_currents, nfp: int) -> float:
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    return 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))


def test_remap_tensor_fourier_index_keeps_cosine_and_shifts_sine():
    assert remap_tensor_fourier_index(0, 2, 16) == 0
    assert remap_tensor_fourier_index(2, 2, 16) == 2
    assert remap_tensor_fourier_index(3, 2, 16) == 17
    assert remap_tensor_fourier_index(4, 2, 16) == 18


def test_upsample_surface_preserves_volume():
    base_curves, base_currents, magnetic_axis, nfp, _biotsavart = get_data("ncsx")
    del base_curves, base_currents
    surface = get_surface(
        "SurfaceXYZTensorFourier",
        True,
        mpol=2,
        ntor=2,
        nphi=7,
        ntheta=7,
        nfp=nfp,
    )
    surface.fit_to_curve(magnetic_axis, 0.1, flip_theta=True)
    padded = upsample_surface_xyz_tensor_fourier(
        surface, mpol=4, ntor=4, nphi=12, ntheta=12
    )
    volume_tol = parity_ladder_tolerances("direct_kernel")
    np.testing.assert_allclose(
        float(Volume(padded).J()),
        float(Volume(surface).J()),
        rtol=float(volume_tol["rtol"]),
        atol=float(volume_tol["atol"]),
    )


def _ncsx_7x7_pair():
    base_curves, base_currents, magnetic_axis, nfp, biotsavart = get_data("ncsx")
    surface = get_surface(
        "SurfaceXYZTensorFourier",
        True,
        mpol=2,
        ntor=2,
        nphi=7,
        ntheta=7,
        nfp=nfp,
    )
    surface.fit_to_curve(magnetic_axis, 0.1, flip_theta=True)
    native_surface = _clone_upstream_surface(surface)
    jax_surface = _clone_upstream_surface(surface)
    native_label = Volume(native_surface)
    jax_label = Volume(jax_surface)
    target = float(native_label.J())
    newton_options = {
        "verbose": False,
        "newton_tol": NESTED_LS_BANANA_NEWTON_TOL,
        "newton_maxiter": NESTED_LS_BANANA_NEWTON_MAXITER,
        "weight_inv_modB": NESTED_LS_WEIGHT_INV_MODB,
    }
    native = BoozerSurface(
        biotsavart,
        native_surface,
        native_label,
        target,
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        options=newton_options,
    )
    jax_boozer = BoozerSurfaceJAX(
        BiotSavartJAX(biotsavart.coils),
        jax_surface,
        jax_label,
        target,
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        options={
            **newton_options,
            "optimizer_backend": "ondevice",
        },
    )
    g0 = _g_from_currents(base_currents, nfp)
    return native, jax_boozer, base_curves, biotsavart, _IOTA0, g0


def _seed_from_native_lbfgs(native, jax_boozer, iota, g0):
    native.need_to_run_code = True
    polished = native.minimize_boozer_penalty_constraints_LBFGS(
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        iota=float(iota),
        G=float(g0),
        tol=1e-10,
        maxiter=400,
        verbose=False,
        weight_inv_modB=NESTED_LS_WEIGHT_INV_MODB,
    )
    if not bool(polished["success"]):
        raise AssertionError("native LBFGS seed for NCSX nested-LS failed.")
    jax_boozer.surface.set_dofs(native.surface.get_dofs())
    return float(polished["iota"]), float(polished["G"])


@pytest.mark.boozer
def test_ncsx_schur_inner_lands_on_seeded_7x7():
    native, jax_boozer, _base_curves, _bs, iota0, g0 = _ncsx_7x7_pair()
    iota, g_value = _seed_from_native_lbfgs(native, jax_boozer, iota0, g0)
    result = run_ncsx_schur_inner(jax_boozer, iota=iota, G=g_value)
    assert result.success
    assert result.coil_delta_inf == 0.0
    np.testing.assert_allclose(result.iota, iota, rtol=1.0e-5, atol=1.0e-5)


@pytest.mark.boozer
def test_ncsx_projected_y_coil_jacobian_matches_frozen_surface_fd():
    import jax.numpy as jnp

    native, jax_boozer, _base_curves, _bs, iota0, g0 = _ncsx_7x7_pair()
    iota, g_value = _seed_from_native_lbfgs(native, jax_boozer, iota0, g0)
    result = run_ncsx_schur_inner(jax_boozer, iota=iota, G=g_value)
    assert result.success
    residual_rt, _objective, _phi = nested_ls_runtime_coil_closures(jax_boozer)
    del _objective, _phi
    coil = np.asarray(jax_boozer.biotsavart.x, dtype=np.float64).reshape(-1)
    y_probe = np.array([float(result.iota), float(result.G)], dtype=np.float64)
    jacobian = np.asarray(
        ncsx_mod._projected_y_coil_jacobian(
            residual_rt,
            result.surface_dofs,
            coil,
            y_probe,
        ),
        dtype=np.float64,
    )
    assert jacobian.shape == (2, coil.size)
    column = int(np.argmax(np.linalg.norm(jacobian, axis=0)))
    step = 1.0e-6 * max(1.0, abs(float(coil[column])))

    def y_at(coil_vector):
        coil_jax = jnp.asarray(coil_vector, dtype=jnp.float64)

        def residual_fn(packed):
            return residual_rt(packed, coil_jax)

        solution = solve_projected_y(residual_fn, result.surface_dofs, y_probe)
        return np.asarray(solution.solution, dtype=np.float64)

    plus = np.array(coil, dtype=np.float64, copy=True)
    minus = np.array(coil, dtype=np.float64, copy=True)
    plus[column] += step
    minus[column] -= step
    finite_difference = (y_at(plus) - y_at(minus)) / (2.0 * step)
    np.testing.assert_allclose(
        jacobian[:, column],
        finite_difference,
        rtol=1.0e-4,
        atol=1.0e-6,
    )


@pytest.mark.boozer
def test_ncsx_outer_value_and_grad_is_finite_on_7x7(monkeypatch):
    native, jax_boozer, base_curves, biotsavart, iota0, g0 = _ncsx_7x7_pair()
    iota, g_value = _seed_from_native_lbfgs(native, jax_boozer, iota0, g0)
    jax_boozer.surface.set_dofs(native.surface.get_dofs())
    curves = [coil.curve for coil in biotsavart.coils]
    problem = prepare_ncsx_nested_ls_problem(
        coils=biotsavart.coils,
        base_curves=base_curves,
        curves=curves,
        surfaces=[clone_surface_xyz_tensor_fourier(native.surface)],
        iotas=[iota],
        g_values=[g_value],
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
    )
    _stub_surfaces_not_self_intersecting(monkeypatch, problem)
    coil = np.asarray(problem.biotsavart.x, dtype=np.float64)
    value, gradient = ncsx_nested_ls_outer_value_and_grad(problem, coil)
    assert np.isfinite(value)
    assert gradient.shape == coil.shape
    assert bool(np.all(np.isfinite(gradient)))
    assert problem.last_inner is not None
    assert problem.last_inner.success
    assert tuple(problem.last_eval_timing) == NCSX_EVAL_TIMING_KEYS
    assert problem.last_eval_timing["total"] > 0.0
    assert problem.last_eval_timing["inner"] > 0.0
    assert problem.last_eval_timing["y_coil_jacobian"] > 0.0
    assert problem.surfaces[0].jax_boozer.options["newton_linear_solver"] == (
        "dense_lu"
    )


@pytest.mark.boozer
def test_ncsx_restore_anchor_discards_poisoned_trial(monkeypatch):
    native, jax_boozer, base_curves, biotsavart, iota0, g0 = _ncsx_7x7_pair()
    iota, g_value = _seed_from_native_lbfgs(native, jax_boozer, iota0, g0)
    jax_boozer.surface.set_dofs(native.surface.get_dofs())
    curves = [coil.curve for coil in biotsavart.coils]
    problem = prepare_ncsx_nested_ls_problem(
        coils=biotsavart.coils,
        base_curves=base_curves,
        curves=curves,
        surfaces=[clone_surface_xyz_tensor_fourier(native.surface)],
        iotas=[iota],
        g_values=[g_value],
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
    )
    state = problem.surfaces[0]
    committed_surface = np.array(state.anchor_surface_dofs, dtype=np.float64, copy=True)
    committed_iota = float(state.anchor_iota)
    committed_g = float(state.anchor_G)
    wrecked = np.array(committed_surface, dtype=np.float64, copy=True)
    wrecked[0] += 1.0
    state.jax_boozer.surface.set_dofs(wrecked)
    state.iota = 99.0
    state.G = -99.0
    restore_ncsx_anchor(problem)
    np.testing.assert_array_equal(
        np.asarray(state.jax_boozer.surface.get_dofs(), dtype=np.float64),
        committed_surface,
    )
    assert state.iota == committed_iota
    assert state.G == committed_g
    state.jax_boozer.surface.set_dofs(wrecked)
    state.iota = 99.0
    _stub_surfaces_not_self_intersecting(monkeypatch, problem)
    coil = np.asarray(problem.biotsavart.x, dtype=np.float64)
    seen: list[tuple[np.ndarray, float, float]] = []
    real_inner = ncsx_mod.run_ncsx_schur_inner

    def spy_inner(jax_boozer, **kwargs):
        seen.append(
            (
                np.array(jax_boozer.surface.get_dofs(), dtype=np.float64, copy=True),
                float(kwargs["iota"]),
                float(kwargs["G"]),
            )
        )
        return real_inner(jax_boozer, **kwargs)

    monkeypatch.setattr(ncsx_mod, "run_ncsx_schur_inner", spy_inner)
    value, gradient = ncsx_nested_ls_outer_value_and_grad(problem, coil)
    assert np.isfinite(value)
    assert bool(np.all(np.isfinite(gradient)))
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0][0], committed_surface)
    assert seen[0][1] == committed_iota
    assert seen[0][2] == committed_g


@pytest.mark.boozer
def test_ncsx_outer_eval_does_not_commit_anchor(monkeypatch):
    native, jax_boozer, base_curves, biotsavart, iota0, g0 = _ncsx_7x7_pair()
    iota, g_value = _seed_from_native_lbfgs(native, jax_boozer, iota0, g0)
    jax_boozer.surface.set_dofs(native.surface.get_dofs())
    curves = [coil.curve for coil in biotsavart.coils]
    problem = prepare_ncsx_nested_ls_problem(
        coils=biotsavart.coils,
        base_curves=base_curves,
        curves=curves,
        surfaces=[clone_surface_xyz_tensor_fourier(native.surface)],
        iotas=[iota],
        g_values=[g_value],
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
    )
    state = problem.surfaces[0]
    committed_iota = float(state.anchor_iota)
    committed_surface = np.array(state.anchor_surface_dofs, dtype=np.float64, copy=True)
    _stub_surfaces_not_self_intersecting(monkeypatch, problem)
    coil = np.asarray(problem.biotsavart.x, dtype=np.float64)
    ncsx_nested_ls_outer_value_and_grad(problem, coil)
    np.testing.assert_array_equal(state.anchor_surface_dofs, committed_surface)
    assert state.anchor_iota == committed_iota
    assert state.anchor_G == float(g_value)


def _ncsx_prepared_7x7():
    native, jax_boozer, base_curves, biotsavart, iota0, g0 = _ncsx_7x7_pair()
    iota, g_value = _seed_from_native_lbfgs(native, jax_boozer, iota0, g0)
    jax_boozer.surface.set_dofs(native.surface.get_dofs())
    curves = [coil.curve for coil in biotsavart.coils]
    problem = prepare_ncsx_nested_ls_problem(
        coils=biotsavart.coils,
        base_curves=base_curves,
        curves=curves,
        surfaces=[clone_surface_xyz_tensor_fourier(native.surface)],
        iotas=[iota],
        g_values=[g_value],
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
    )
    return problem


def _stub_surfaces_not_self_intersecting(monkeypatch, problem):
    for surface_state in problem.surfaces:
        if surface_state.jax_boozer is not None:
            monkeypatch.setattr(
                surface_state.jax_boozer.surface,
                "is_self_intersecting",
                lambda *args, **kwargs: False,
            )
        if surface_state.native is not None:
            monkeypatch.setattr(
                surface_state.native.surface,
                "is_self_intersecting",
                lambda *args, **kwargs: False,
            )


def _assert_anchor_restored(problem, *, surface, iota, g_value):
    state = problem.surfaces[0]
    np.testing.assert_array_equal(
        np.asarray(state.jax_boozer.surface.get_dofs(), dtype=np.float64),
        surface,
    )
    assert state.iota == iota
    assert state.G == g_value
    np.testing.assert_array_equal(state.anchor_surface_dofs, surface)
    assert state.anchor_iota == iota
    assert state.anchor_G == g_value


@pytest.mark.boozer
def test_ncsx_failed_inner_restores_persistable_poison(monkeypatch):
    problem = _ncsx_prepared_7x7()
    state = problem.surfaces[0]
    committed_surface = np.array(state.anchor_surface_dofs, dtype=np.float64, copy=True)
    committed_iota = float(state.anchor_iota)
    committed_g = float(state.anchor_G)
    coil = np.asarray(problem.biotsavart.x, dtype=np.float64)

    def poison_and_fail(jax_boozer, **_kwargs):
        wrecked = np.array(jax_boozer.surface.get_dofs(), dtype=np.float64, copy=True)
        wrecked[0] += 1.0
        jax_boozer.surface.set_dofs(wrecked)
        return SimpleNamespace(
            success=False,
            iteration_count=3,
            reduced_gradient=np.array([510.0], dtype=np.float64),
            exit_status="failed",
            iota=99.0,
            G=-99.0,
        )

    monkeypatch.setattr(
        "simsopt_jax_adapters.geo.nested_ls_ncsx.run_ncsx_schur_inner",
        poison_and_fail,
    )
    with pytest.raises(NcsxNestedLsInnerSolveFailed):
        ncsx_nested_ls_outer_value_and_grad(problem, coil)
    _assert_anchor_restored(
        problem,
        surface=committed_surface,
        iota=committed_iota,
        g_value=committed_g,
    )

    def poison_and_jump(jax_boozer, *, iota, G, **_kwargs):
        wrecked = np.array(jax_boozer.surface.get_dofs(), dtype=np.float64, copy=True)
        wrecked[0] += 1.0
        jax_boozer.surface.set_dofs(wrecked)
        return SimpleNamespace(
            success=True,
            iteration_count=4,
            reduced_gradient=np.zeros(1, dtype=np.float64),
            exit_status="converged",
            iota=float(iota) + 0.2,
            G=float(G),
        )

    monkeypatch.setattr(
        "simsopt_jax_adapters.geo.nested_ls_ncsx.run_ncsx_schur_inner",
        poison_and_jump,
    )
    with pytest.raises(NcsxNestedLsBranchJump):
        ncsx_nested_ls_outer_value_and_grad(problem, coil)
    _assert_anchor_restored(
        problem,
        surface=committed_surface,
        iota=committed_iota,
        g_value=committed_g,
    )


@pytest.mark.boozer
def test_ncsx_self_intersecting_trial_restores_anchor(monkeypatch):
    problem = _ncsx_prepared_7x7()
    state = problem.surfaces[0]
    committed_surface = np.array(state.anchor_surface_dofs, dtype=np.float64, copy=True)
    committed_iota = float(state.anchor_iota)
    committed_g = float(state.anchor_G)
    coil = np.asarray(problem.biotsavart.x, dtype=np.float64)
    monkeypatch.setattr(
        state.jax_boozer.surface,
        "is_self_intersecting",
        lambda *args, **kwargs: True,
    )
    with pytest.raises(NcsxNestedLsSelfIntersecting):
        ncsx_nested_ls_outer_value_and_grad(problem, coil)
    _assert_anchor_restored(
        problem,
        surface=committed_surface,
        iota=committed_iota,
        g_value=committed_g,
    )


@pytest.mark.boozer
def test_ncsx_identical_two_surface_mean_matches_one_surface(monkeypatch):
    native, jax_boozer, base_curves, biotsavart, iota0, g0 = _ncsx_7x7_pair()
    iota, g_value = _seed_from_native_lbfgs(native, jax_boozer, iota0, g0)
    jax_boozer.surface.set_dofs(native.surface.get_dofs())
    curves = [coil.curve for coil in biotsavart.coils]
    surface = clone_surface_xyz_tensor_fourier(native.surface)
    kwargs = dict(
        coils=biotsavart.coils,
        base_curves=base_curves,
        curves=curves,
        iotas=[iota],
        g_values=[g_value],
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
    )
    one = prepare_ncsx_nested_ls_problem(
        surfaces=[clone_surface_xyz_tensor_fourier(surface)],
        **kwargs,
    )
    two = prepare_ncsx_nested_ls_problem(
        surfaces=[
            clone_surface_xyz_tensor_fourier(surface),
            clone_surface_xyz_tensor_fourier(surface),
        ],
        iotas=[iota, iota],
        g_values=[g_value, g_value],
        coils=biotsavart.coils,
        base_curves=base_curves,
        curves=curves,
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
    )
    assert two.surfaces[0].radius_scale == 0.0
    assert two.surfaces[1].radius_scale == NCSX_MR_WEIGHT
    _stub_surfaces_not_self_intersecting(monkeypatch, one)
    _stub_surfaces_not_self_intersecting(monkeypatch, two)
    coil = np.asarray(one.biotsavart.x, dtype=np.float64)
    value_one, grad_one = ncsx_nested_ls_outer_value_and_grad(one, coil)
    value_two, grad_two = ncsx_nested_ls_outer_value_and_grad(two, coil)
    np.testing.assert_allclose(value_two, value_one, rtol=1.0e-8, atol=1.0e-8)
    np.testing.assert_allclose(grad_two, grad_one, rtol=1.0e-7, atol=1.0e-7)


def test_ncsx_prepare_rejects_empty_surfaces():
    with pytest.raises(ValueError, match="at least one surface"):
        prepare_ncsx_nested_ls_problem(
            coils=[],
            base_curves=[],
            curves=[],
            surfaces=[],
            iotas=[],
            g_values=[],
            constraint_weight=1.0,
        )


def test_ncsx_eval_timing_summary_means_match_sums():
    zeros = empty_ncsx_eval_timing()
    assert tuple(zeros) == NCSX_EVAL_TIMING_KEYS
    empty = summarize_ncsx_eval_timings([])
    assert empty["n"] == 0
    first = empty_ncsx_eval_timing()
    first["inner"] = 4.0
    first["total"] = 10.0
    second = empty_ncsx_eval_timing()
    second["inner"] = 6.0
    second["total"] = 14.0
    summary = summarize_ncsx_eval_timings([first, second])
    assert summary["n"] == 2
    assert summary["sum"]["inner"] == 10.0
    assert summary["mean"]["inner"] == 5.0
    assert summary["mean"]["total"] == 12.0


def _ncsx_prepared_7x7_native():
    native, jax_boozer, base_curves, biotsavart, iota0, g0 = _ncsx_7x7_pair()
    iota, g_value = _seed_from_native_lbfgs(native, jax_boozer, iota0, g0)
    curves = [coil.curve for coil in biotsavart.coils]
    return prepare_ncsx_nested_ls_problem(
        coils=biotsavart.coils,
        base_curves=base_curves,
        curves=curves,
        surfaces=[clone_surface_xyz_tensor_fourier(native.surface)],
        iotas=[iota],
        g_values=[g_value],
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        include_native=True,
    )


@pytest.mark.boozer
def test_ncsx_native_outer_value_and_grad_is_finite_on_7x7(monkeypatch):
    problem = _ncsx_prepared_7x7_native()
    assert problem.native_objective is not None
    assert problem.surfaces[0].native is not None
    _stub_surfaces_not_self_intersecting(monkeypatch, problem)
    coil = np.asarray(problem.native_biotsavart.x, dtype=np.float64)
    value, gradient = ncsx_native_outer_value_and_grad(problem, coil)
    assert np.isfinite(value)
    assert gradient.shape == coil.shape
    assert bool(np.all(np.isfinite(gradient)))
    assert tuple(problem.last_eval_timing) == NCSX_EVAL_TIMING_KEYS
    assert problem.last_eval_timing["inner"] > 0.0
    assert problem.last_eval_timing["direct_partials"] > 0.0
    assert problem.last_eval_timing["y_coil_jacobian"] == 0.0
    assert problem.last_native_inner is not None
    assert bool(problem.last_native_inner["success"])


@pytest.mark.boozer
def test_ncsx_native_failed_inner_restores_persistable_poison(monkeypatch):
    problem = _ncsx_prepared_7x7_native()
    state = problem.surfaces[0]
    committed_surface = np.array(state.anchor_surface_dofs, dtype=np.float64, copy=True)
    committed_iota = float(state.anchor_iota)
    committed_g = float(state.anchor_G)
    coil = np.asarray(problem.native_biotsavart.x, dtype=np.float64)

    def poison_and_fail(_iota, _g=None):
        wrecked = np.array(state.native.surface.get_dofs(), dtype=np.float64, copy=True)
        wrecked[0] += 1.0
        state.native.surface.set_dofs(wrecked)
        return {
            "success": False,
            "iter": 3,
            "iota": 99.0,
            "G": -99.0,
            "jacobian": np.array([510.0], dtype=np.float64),
        }

    monkeypatch.setattr(state.native, "run_code", poison_and_fail)
    with pytest.raises(NcsxNestedLsInnerSolveFailed):
        ncsx_native_outer_value_and_grad(problem, coil)
    np.testing.assert_array_equal(
        np.asarray(state.native.surface.get_dofs(), dtype=np.float64),
        committed_surface,
    )
    _assert_anchor_restored(
        problem,
        surface=committed_surface,
        iota=committed_iota,
        g_value=committed_g,
    )


def test_ncsx_problem_from_native_boozers_rejects_empty():
    with pytest.raises(ValueError, match="at least one surface"):
        ncsx_problem_from_native_boozers(
            [],
            iotas=[],
            g_values=[],
            base_curves=[],
            curves=[],
        )
