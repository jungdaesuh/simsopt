"""Mixed lowering must not embed dense surface-scatter matrices."""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field.coil import Current, coils_via_symmetries
from simsopt.geo import SurfaceXYZTensorFourier, Volume
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.geo.surfacerzfourier import SurfaceRZFourier

import simsopt_jax_adapters.geo.surface_objectives_traceable as traceable_module
from conftest import enable_non_strict_jax_backend
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX


_CONSTANT_TENSOR_RE = re.compile(
    r"stablehlo\.constant[^\n]*?tensor<([0-9x]*)x?(f32|f64)>"
)


def _large_constant_dtype_census(stablehlo_text: str, min_elems: int):
    census: dict[str, list[int]] = {"f32": [], "f64": []}
    for shape_str, dtype_name in _CONSTANT_TENSOR_RE.findall(stablehlo_text):
        elems = 1
        for part in shape_str.split("x"):
            if part:
                elems *= int(part)
        if elems >= min_elems:
            census[dtype_name].append(elems)
    return census


def _solved_small_boozer_fixture():
    ncoils, nfp, stellsym = 2, 2, True
    base_curves = create_equally_spaced_curves(
        ncoils, nfp, stellsym=stellsym, R0=1.0, R1=0.5, order=3
    )
    base_currents = [Current(1.0e5) for _ in range(ncoils)]
    for current in base_currents:
        current.fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)

    mpol = ntor = 2
    nphi, ntheta = 2 * ntor + 1, 2 * mpol + 1
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=stellsym,
        nfp=nfp,
        quadpoints_phi=np.linspace(0, 1.0 / nfp, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0, 1.0, ntheta, endpoint=False),
    )
    surface.set_dofs(np.zeros_like(surface.get_dofs()))
    reference_surface = SurfaceRZFourier(
        nfp=nfp,
        stellsym=stellsym,
        mpol=1,
        ntor=0,
        quadpoints_phi=surface.quadpoints_phi,
        quadpoints_theta=surface.quadpoints_theta,
    )
    reference_surface.set_rc(0, 0, 1.0)
    reference_surface.set_rc(1, 0, 0.15)
    reference_surface.set_zs(1, 0, 0.15)
    surface.least_squares_fit(reference_surface.gamma())

    bs_jax = BiotSavartJAX(coils)
    volume = Volume(surface)
    mu0 = 4.0 * np.pi * 1.0e-7
    G0 = mu0 * sum(abs(coil.current.get_value()) for coil in coils)
    booz_jax = BoozerSurfaceJAX(
        bs_jax,
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
    result = booz_jax.run_code(0.3, G0)
    assert result is not None and result.get("success", False)
    booz_jax.res["linearization_kind"] = "hessian"
    return booz_jax, bs_jax


def test_mixed_bundle_lowering_uses_matrix_free_surface_scatter(
    monkeypatch,
    request,
) -> None:
    booz_jax, bs_jax = _solved_small_boozer_fixture()
    iota_target = jnp.asarray(booz_jax.res["iota"], dtype=jnp.float64)
    coil_dofs = jnp.asarray(np.asarray(bs_jax.x).copy(), dtype=jnp.float64)

    enable_non_strict_jax_backend(
        monkeypatch, request, mode="jax_cpu_fast", precision="mixed"
    )
    state = traceable_module._build_traceable_objective_state(
        booz_jax,
        bs_jax,
        iota_target,
    )
    bundle = traceable_module._build_traceable_objective_compiled_bundle_from_state(
        booz_jax,
        state,
    )
    stablehlo_text = (
        jax.jit(bundle["compiled_value_and_grad_for"]).lower(coil_dofs).as_text()
    )

    assert "stablehlo.scatter" in stablehlo_text
    census = _large_constant_dtype_census(stablehlo_text, min_elems=2048)
    assert not census["f64"], sorted(census["f64"])
    assert not census["f32"], sorted(census["f32"])
