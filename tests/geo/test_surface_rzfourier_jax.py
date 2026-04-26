import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from conftest import host_array, host_scalar, parity_default_device, parity_rng

from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surface import Surface
from simsopt.geo.boozer_residual_jax import _surface_geometry_from_dofs
from simsopt.jax_core import (
    SurfaceRZFourierSpec,
    make_surface_rzfourier_spec,
    surface_rz_fourier_area_from_dofs,
    surface_rz_fourier_area_from_spec,
    surface_rz_fourier_dofs_from_spec,
    surface_rz_fourier_gamma_from_dofs,
    surface_rz_fourier_gamma_from_spec,
    surface_rz_fourier_gammadash1_from_dofs,
    surface_rz_fourier_gammadash1_from_spec,
    surface_rz_fourier_gammadash2_from_dofs,
    surface_rz_fourier_gammadash2_from_spec,
    surface_rz_fourier_normal_from_dofs,
    surface_rz_fourier_normal_from_spec,
    surface_rz_fourier_dnormal_from_dofs,
    surface_rz_fourier_spec_from_dofs,
    surface_rz_fourier_unitnormal_from_dofs,
    surface_rz_fourier_unitnormal_from_spec,
    surface_rz_fourier_dunitnormal_from_dofs,
    surface_rz_fourier_volume_from_dofs,
    surface_rz_fourier_volume_from_spec,
)
from simsopt.jax_core.surface_rzfourier import surface_rz_fourier_geometry_from_spec

TEST_DIR = Path(__file__).parent / ".." / "test_files"
SURFACE_RZFOURIER_VOLUME_ATOL = 1e-8

def _make_surface(*, stellsym: bool) -> SurfaceRZFourier:
    rng = parity_rng(7 if stellsym else 11)
    surface = SurfaceRZFourier.from_nphi_ntheta(
        nfp=2,
        stellsym=stellsym,
        mpol=2,
        ntor=1,
        nphi=9,
        ntheta=10,
        range="field period",
    )
    surface.rc[:, :] = rng.normal(scale=0.05, size=surface.rc.shape)
    surface.zs[:, :] = rng.normal(scale=0.05, size=surface.zs.shape)
    surface.rc[0, surface.ntor] = 1.2
    surface.rc[1, surface.ntor] += 0.15
    surface.zs[1, surface.ntor] += 0.08
    surface.rc[0, : surface.ntor] = 0.0
    surface.zs[0, : surface.ntor + 1] = 0.0
    if not stellsym:
        surface.rs[:, :] = rng.normal(scale=0.03, size=surface.rs.shape)
        surface.zc[:, :] = rng.normal(scale=0.03, size=surface.zc.shape)
        surface.rs[0, : surface.ntor + 1] = 0.0
        surface.zc[0, : surface.ntor] = 0.0
    surface.local_full_x = surface.get_dofs()
    return surface


@pytest.fixture(autouse=True)
def _parity_device_scope(parity_lane):
    with parity_default_device(parity_lane):
        yield


def _surface_spec_from_surface(surface: SurfaceRZFourier):
    return surface_rz_fourier_spec_from_dofs(
        surface.get_dofs(),
        quadpoints_phi=surface.quadpoints_phi,
        quadpoints_theta=surface.quadpoints_theta,
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
    )


def _scalar_geometry_from_spec(spec: SurfaceRZFourierSpec):
    return (
        surface_rz_fourier_gamma_from_spec(spec),
        surface_rz_fourier_gammadash1_from_spec(spec),
        surface_rz_fourier_gammadash2_from_spec(spec),
    )


def _geometry_vector_from_dofs(surface: SurfaceRZFourier, dofs, evaluator):
    spec = surface_rz_fourier_spec_from_dofs(
        dofs,
        quadpoints_phi=surface.quadpoints_phi,
        quadpoints_theta=surface.quadpoints_theta,
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
    )
    return jnp.concatenate([jnp.ravel(part) for part in evaluator(spec)])


def _hlo_stats(text: str) -> dict[str, int]:
    return {
        "cosine": len(re.findall(r"\bcosine(?:\(|\b)", text)),
        "sine": len(re.findall(r"\bsine(?:\(|\b)", text)),
        "reduce": len(re.findall(r"\breduce(?:\(|\b)", text)),
        "fusion": len(re.findall(r"\bfusion(?:\(|\b)", text)),
        "line_count": text.count("\n") + 1,
    }


def _compiled_hlo_stats(fn, *args) -> dict[str, int]:
    return _hlo_stats(jax.jit(fn).lower(*args).compile().as_text())


def _lowered_hlo_stats(fn, *args) -> dict[str, int]:
    return _hlo_stats(jax.jit(fn).lower(*args).as_text())


def _make_hlo_probe_surface() -> SurfaceRZFourier:
    surface = SurfaceRZFourier.from_nphi_ntheta(
        nfp=2,
        stellsym=True,
        mpol=8,
        ntor=6,
        nphi=17,
        ntheta=18,
        range="field period",
    )
    rng = parity_rng(1729)
    surface.rc[:, :] = rng.normal(scale=0.02, size=surface.rc.shape)
    surface.zs[:, :] = rng.normal(scale=0.02, size=surface.zs.shape)
    surface.rc[0, surface.ntor] = 1.2
    surface.rc[1, surface.ntor] += 0.15
    surface.zs[1, surface.ntor] += 0.08
    surface.local_full_x = surface.get_dofs()
    return surface


def _assert_dofs_round_trip(surface: SurfaceRZFourier) -> None:
    spec = _surface_spec_from_surface(surface)
    np.testing.assert_allclose(
        np.asarray(surface_rz_fourier_dofs_from_spec(spec)),
        np.asarray(surface.get_dofs()),
        rtol=0.0,
        atol=1e-12,
    )


def _assert_surface_parity(surface: SurfaceRZFourier) -> None:
    spec = surface.surface_spec()
    dofs = surface.get_dofs()
    assert isinstance(spec, SurfaceRZFourierSpec)

    gamma_jax = host_array(surface_rz_fourier_gamma_from_spec(spec))
    gd1_jax = host_array(surface_rz_fourier_gammadash1_from_spec(spec))
    gd2_jax = host_array(surface_rz_fourier_gammadash2_from_spec(spec))
    gamma_fused, gd1_fused, gd2_fused = surface_rz_fourier_geometry_from_spec(spec)
    gamma_fused = host_array(gamma_fused)
    gd1_fused = host_array(gd1_fused)
    gd2_fused = host_array(gd2_fused)
    normal_jax = host_array(surface_rz_fourier_normal_from_spec(spec))
    unitnormal_jax = host_array(surface_rz_fourier_unitnormal_from_spec(spec))
    gamma_from_dofs = host_array(surface_rz_fourier_gamma_from_dofs(spec, dofs))
    gd1_from_dofs = host_array(surface_rz_fourier_gammadash1_from_dofs(spec, dofs))
    gd2_from_dofs = host_array(surface_rz_fourier_gammadash2_from_dofs(spec, dofs))
    normal_from_dofs = host_array(surface_rz_fourier_normal_from_dofs(spec, dofs))
    unitnormal_from_dofs = host_array(surface_rz_fourier_unitnormal_from_dofs(spec, dofs))

    np.testing.assert_allclose(gamma_jax, surface.gamma(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gd1_jax, surface.gammadash1(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gd2_jax, surface.gammadash2(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gamma_fused, surface.gamma(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gd1_fused, surface.gammadash1(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gd2_fused, surface.gammadash2(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(normal_jax, surface.normal(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        unitnormal_jax, surface.unitnormal(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(gamma_from_dofs, surface.gamma(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        gd1_from_dofs, surface.gammadash1(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        gd2_from_dofs, surface.gammadash2(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        normal_from_dofs, surface.normal(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        unitnormal_from_dofs, surface.unitnormal(), rtol=1e-12, atol=1e-12
    )

    np.testing.assert_allclose(
        host_array(surface.gamma_jax()),
        surface.gamma(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        host_array(surface.gamma_jax(dofs)),
        surface.gamma(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        host_array(surface.normal_jax()),
        surface.normal(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        host_array(surface.normal_jax(dofs)),
        surface.normal(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        host_array(surface.unitnormal_jax()),
        surface.unitnormal(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        host_array(surface.unitnormal_jax(dofs)),
        surface.unitnormal(),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        host_scalar(surface_rz_fourier_area_from_spec(spec)),
        surface.area(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        host_scalar(surface_rz_fourier_area_from_dofs(spec, dofs)),
        surface.area(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        host_scalar(surface_rz_fourier_volume_from_spec(spec)),
        surface.volume(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        host_scalar(surface_rz_fourier_volume_from_dofs(spec, dofs)),
        surface.volume(),
        rtol=1e-12,
        atol=1e-12,
    )


def _assert_loaded_surface_object_api_parity(
    surface: SurfaceRZFourier, *, expected_volume: float
) -> None:
    np.testing.assert_allclose(surface.x, surface.get_dofs(), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        surface.volume(),
        expected_volume,
        rtol=0.0,
        atol=SURFACE_RZFOURIER_VOLUME_ATOL,
    )
    _assert_surface_parity(surface)


def _surface_copy_variants(surface: SurfaceRZFourier) -> tuple[SurfaceRZFourier, ...]:
    return (
        surface.copy(
            quadpoints_phi=Surface.get_phi_quadpoints(nphi=100, range="field period")
        ),
        surface.copy(quadpoints_theta=Surface.get_theta_quadpoints(ntheta=50)),
        surface.copy(ntheta=42),
        surface.copy(nphi=17),
        surface.copy(range="field period"),
        surface.copy(nfp=10),
        surface.copy(mpol=5, ntor=6),
        surface.copy(stellsym=False),
    )


def test_surface_rzfourier_jax_parity_stellsym():
    _assert_surface_parity(_make_surface(stellsym=True))


def test_surface_rzfourier_jax_parity_non_stellsym():
    _assert_surface_parity(_make_surface(stellsym=False))


def test_surface_rzfourier_spec_is_jittable():
    spec = make_surface_rzfourier_spec(
        rc=np.asarray([[1.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        zs=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        quadpoints_phi=np.linspace(0.0, 0.5, 4, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False),
        nfp=2,
        stellsym=True,
    )
    gamma = jax.jit(surface_rz_fourier_gamma_from_spec)(spec)
    normal = jax.jit(surface_rz_fourier_normal_from_spec)(spec)
    assert gamma.shape == (4, 5, 3)
    assert normal.shape == (4, 5, 3)


@pytest.mark.parametrize("stellsym", [True, False])
def test_surface_rzfourier_geometry_from_spec_matches_scalar_composition(stellsym):
    spec = _make_surface(stellsym=stellsym).surface_spec()
    fused_geometry = surface_rz_fourier_geometry_from_spec(spec)
    scalar_geometry = _scalar_geometry_from_spec(spec)

    for fused_part, scalar_part in zip(fused_geometry, scalar_geometry):
        np.testing.assert_allclose(
            host_array(fused_part),
            host_array(scalar_part),
            rtol=1e-12,
            atol=1e-12,
        )


def test_surface_rzfourier_geometry_jvp_matches_scalar_composition():
    surface = _make_surface(stellsym=False)
    dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)
    tangent = jnp.asarray(
        parity_rng(19).normal(scale=0.01, size=dofs.shape),
        dtype=jnp.float64,
    )
    fused_value, fused_tangent = jax.jvp(
        lambda x: _geometry_vector_from_dofs(
            surface,
            x,
            surface_rz_fourier_geometry_from_spec,
        ),
        (dofs,),
        (tangent,),
    )
    scalar_value, scalar_tangent = jax.jvp(
        lambda x: _geometry_vector_from_dofs(surface, x, _scalar_geometry_from_spec),
        (dofs,),
        (tangent,),
    )

    np.testing.assert_allclose(
        host_array(fused_value),
        host_array(scalar_value),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        host_array(fused_tangent),
        host_array(scalar_tangent),
        rtol=1e-12,
        atol=1e-12,
    )


def test_surface_rzfourier_geometry_vjp_cotangent_matches_scalar_composition():
    surface = _make_surface(stellsym=False)
    dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)
    cotangent_shape = _geometry_vector_from_dofs(
        surface,
        dofs,
        surface_rz_fourier_geometry_from_spec,
    ).shape
    cotangent = jnp.asarray(
        parity_rng(23).normal(scale=0.01, size=cotangent_shape),
        dtype=jnp.float64,
    )
    _, fused_pullback = jax.vjp(
        lambda x: _geometry_vector_from_dofs(
            surface,
            x,
            surface_rz_fourier_geometry_from_spec,
        ),
        dofs,
    )
    _, scalar_pullback = jax.vjp(
        lambda x: _geometry_vector_from_dofs(surface, x, _scalar_geometry_from_spec),
        dofs,
    )

    np.testing.assert_allclose(
        host_array(fused_pullback(cotangent)[0]),
        host_array(scalar_pullback(cotangent)[0]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_surface_rzfourier_geometry_jacfwd_matches_scalar_composition():
    surface = _make_surface(stellsym=False)
    dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)

    fused_jacobian = jax.jacfwd(
        lambda x: _geometry_vector_from_dofs(
            surface,
            x,
            surface_rz_fourier_geometry_from_spec,
        )
    )(dofs)
    scalar_jacobian = jax.jacfwd(
        lambda x: _geometry_vector_from_dofs(surface, x, _scalar_geometry_from_spec)
    )(dofs)

    np.testing.assert_allclose(
        host_array(fused_jacobian),
        host_array(scalar_jacobian),
        rtol=1e-12,
        atol=1e-12,
    )


def test_surface_rzfourier_geometry_hessian_trace_smoke():
    surface = _make_surface(stellsym=True)
    dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)

    hessian = jax.hessian(
        lambda x: jnp.sum(
            _geometry_vector_from_dofs(
                surface,
                x,
                surface_rz_fourier_geometry_from_spec,
            )
            ** 2
        )
    )(dofs)

    assert hessian.shape == (dofs.size, dofs.size)
    assert np.all(np.isfinite(host_array(hessian)))


def test_surface_rzfourier_geometry_allows_strict_transfer_guard():
    spec = _make_surface(stellsym=True).surface_spec()
    geometry_fn = jax.jit(surface_rz_fourier_geometry_from_spec)

    with jax.transfer_guard("disallow"):
        gamma, xphi, xtheta = geometry_fn(spec)
        gamma.block_until_ready()
        xphi.block_until_ready()
        xtheta.block_until_ready()


def test_surface_rzfourier_fused_geometry_reduces_hlo_work():
    surface = _make_hlo_probe_surface()
    dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)
    scalar_geometry = lambda x: _geometry_vector_from_dofs(
        surface,
        x,
        _scalar_geometry_from_spec,
    )
    fused_geometry = lambda x: _geometry_vector_from_dofs(
        surface,
        x,
        surface_rz_fourier_geometry_from_spec,
    )
    scalar_lowered_stats = _lowered_hlo_stats(scalar_geometry, dofs)
    fused_lowered_stats = _lowered_hlo_stats(fused_geometry, dofs)
    scalar_optimized_stats = _compiled_hlo_stats(scalar_geometry, dofs)
    fused_optimized_stats = _compiled_hlo_stats(fused_geometry, dofs)

    assert fused_lowered_stats["cosine"] < scalar_lowered_stats["cosine"]
    assert fused_lowered_stats["sine"] < scalar_lowered_stats["sine"]
    assert fused_lowered_stats["reduce"] < scalar_lowered_stats["reduce"]
    assert fused_optimized_stats["line_count"] < scalar_optimized_stats["line_count"]
    assert fused_optimized_stats["cosine"] <= scalar_optimized_stats["cosine"]
    assert fused_optimized_stats["sine"] <= scalar_optimized_stats["sine"]
    assert fused_optimized_stats["reduce"] <= scalar_optimized_stats["reduce"]


def test_surface_rzfourier_scalar_gamma_hlo_stays_single_output():
    surface = _make_surface(stellsym=True)
    dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)

    gamma_stats = _compiled_hlo_stats(
        lambda x: surface_rz_fourier_gamma_from_dofs(surface.surface_spec(), x),
        dofs,
    )
    geometry_stats = _compiled_hlo_stats(
        lambda x: _geometry_vector_from_dofs(
            surface,
            x,
            surface_rz_fourier_geometry_from_spec,
        ),
        dofs,
    )
    gamma_jacfwd_stats = _compiled_hlo_stats(
        lambda x: jax.jacfwd(
            lambda y: jnp.ravel(
                surface_rz_fourier_gamma_from_dofs(surface.surface_spec(), y)
            )
        )(x),
        dofs,
    )
    geometry_jacfwd_stats = _compiled_hlo_stats(
        lambda x: jax.jacfwd(
            lambda y: _geometry_vector_from_dofs(
                surface,
                y,
                surface_rz_fourier_geometry_from_spec,
            )
        )(x),
        dofs,
    )

    assert gamma_stats["line_count"] < geometry_stats["line_count"]
    assert gamma_stats["cosine"] <= geometry_stats["cosine"]
    assert gamma_stats["sine"] <= geometry_stats["sine"]
    assert gamma_jacfwd_stats["line_count"] < geometry_jacfwd_stats["line_count"]


def test_surface_rz_geometry_hlo_probe_entrypoint_uses_local_package(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    output_json = tmp_path / "surface_rz_geometry_hlo_probe.json"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "benchmarks" / "surface_rz_geometry_hlo_probe.py"),
            "--platform",
            "cpu",
            "--mpol",
            "1",
            "--ntor",
            "1",
            "--nphi",
            "5",
            "--ntheta",
            "6",
            "--repeat",
            "1",
            "--warmup",
            "0",
            "--output-json",
            str(output_json),
        ],
        cwd=repo_root,
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["runtime"]["backend"] == "cpu"
    assert payload["shape"]["mpol"] == 1
    assert payload["comparison"]["hlo_gate_passed"] is True
    assert payload["comparison"]["scalar_api_hlo_guard_passed"] is True
    assert payload["comparison"]["fused_lowered_trig_reduce_counts_lower"] is True


def test_surface_rzfourier_geometry_avoids_jnp_arange(monkeypatch):
    import simsopt.jax_core.surface_rzfourier as sr_jax

    def _fail(*_args, **_kwargs):
        raise AssertionError("surface_rzfourier geometry should not call jnp.arange")

    spec = make_surface_rzfourier_spec(
        rc=np.asarray([[1.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        zs=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        quadpoints_phi=np.linspace(0.0, 0.5, 4, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False),
        nfp=2,
        stellsym=True,
    )
    monkeypatch.setattr(sr_jax.jnp, "arange", _fail)

    gamma = sr_jax.surface_rz_fourier_gamma_from_spec(spec)
    normal = sr_jax.surface_rz_fourier_normal_from_spec(spec)

    assert gamma.shape == (4, 5, 3)
    assert normal.shape == (4, 5, 3)


def test_surface_rzfourier_unitnormal_degenerate_surface_stays_finite():
    spec = make_surface_rzfourier_spec(
        rc=np.zeros((2, 1), dtype=np.float64),
        zs=np.zeros((2, 1), dtype=np.float64),
        quadpoints_phi=np.linspace(0.0, 0.5, 4, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False),
        nfp=2,
        stellsym=True,
    )
    unitnormal = host_array(surface_rz_fourier_unitnormal_from_spec(spec))
    assert np.all(np.isfinite(unitnormal))
    np.testing.assert_array_equal(unitnormal, np.zeros_like(unitnormal))


def test_surface_rzfourier_geometry_from_dofs_matches_boozer_hot_path():
    surface = _make_surface(stellsym=False)
    dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)
    gamma, xphi, xtheta = _surface_geometry_from_dofs(
        dofs,
        jnp.asarray(surface.quadpoints_phi),
        jnp.asarray(surface.quadpoints_theta),
        surface.mpol,
        surface.ntor,
        surface.nfp,
        surface.stellsym,
        None,
        surface_kind="rzfourier",
    )
    np.testing.assert_allclose(
        host_array(gamma), surface.gamma(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        host_array(xphi), surface.gammadash1(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        host_array(xtheta), surface.gammadash2(), rtol=1e-12, atol=1e-12
    )


def _assert_surface_jacobian_parity(surface: SurfaceRZFourier) -> None:
    dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)
    spec = surface.surface_spec()

    normal_jacobian = host_array(surface_rz_fourier_dnormal_from_dofs(spec, dofs))
    unitnormal_jacobian = host_array(
        surface_rz_fourier_dunitnormal_from_dofs(spec, dofs)
    )

    np.testing.assert_allclose(
        normal_jacobian,
        np.asarray(surface.dnormal_by_dcoeff()),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        unitnormal_jacobian,
        np.asarray(surface.dunitnormal_by_dcoeff()),
        rtol=1e-9,
        atol=1e-9,
    )


def test_surface_rzfourier_jax_jacobian_parity_stellsym():
    _assert_surface_jacobian_parity(_make_surface(stellsym=True))


def test_surface_rzfourier_jax_jacobian_parity_non_stellsym():
    _assert_surface_jacobian_parity(_make_surface(stellsym=False))


def _assert_area_volume_gradient_parity(surface: SurfaceRZFourier) -> None:
    dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)
    spec = surface.surface_spec()

    area_grad = np.asarray(
        host_array(jax.grad(lambda x: surface_rz_fourier_area_from_dofs(spec, x))(dofs))
    )
    volume_grad = np.asarray(
        host_array(jax.grad(lambda x: surface_rz_fourier_volume_from_dofs(spec, x))(dofs))
    )

    np.testing.assert_allclose(
        area_grad,
        np.asarray(surface.darea_by_dcoeff()),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        volume_grad,
        np.asarray(surface.dvolume_by_dcoeff()),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(surface.darea_by_dcoeff_jax(dofs)),
        area_grad,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(surface.dvolume_by_dcoeff_jax(dofs)),
        volume_grad,
        rtol=1e-12,
        atol=1e-12,
    )


def test_surface_rzfourier_area_volume_gradient_parity_stellsym():
    _assert_area_volume_gradient_parity(_make_surface(stellsym=True))


def test_surface_rzfourier_area_volume_gradient_parity_non_stellsym():
    _assert_area_volume_gradient_parity(_make_surface(stellsym=False))


def test_surface_rzfourier_spec_from_dofs_round_trip():
    surface = _make_surface(stellsym=False)
    spec = _surface_spec_from_surface(surface)
    np.testing.assert_allclose(
        np.asarray(spec.rc), np.asarray(surface.rc), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(spec.rs), np.asarray(surface.rs), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(spec.zc), np.asarray(surface.zc), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(spec.zs), np.asarray(surface.zs), rtol=0.0, atol=1e-12
    )


def test_surface_rzfourier_dofs_round_trip_stellsym():
    _assert_dofs_round_trip(_make_surface(stellsym=True))


def test_surface_rzfourier_dofs_round_trip_non_stellsym():
    _assert_dofs_round_trip(_make_surface(stellsym=False))


@pytest.mark.parametrize(
    ("filename", "s_value", "expected_volume"),
    [
        ("wout_li383_low_res_reference.nc", 1.0, 2.98138727016329),
        ("wout_LandremanSenguptaPlunk_section5p3_reference.nc", 1.0, 0.199228326859097),
    ],
)
def test_surface_rzfourier_from_wout_object_api_parity(
    filename: str, s_value: float, expected_volume: float
) -> None:
    surface = SurfaceRZFourier.from_wout(TEST_DIR / filename, s=s_value)
    _assert_loaded_surface_object_api_parity(
        surface, expected_volume=expected_volume
    )


@pytest.mark.parametrize(
    ("filename", "expected_volume"),
    [
        ("input.li383_low_res", 2.97871721453671),
        ("input.LandremanSenguptaPlunk_section5p3", 0.199228326303124),
    ],
)
def test_surface_rzfourier_from_vmec_input_object_api_parity(
    filename: str, expected_volume: float
) -> None:
    surface = SurfaceRZFourier.from_vmec_input(TEST_DIR / filename)
    _assert_loaded_surface_object_api_parity(
        surface, expected_volume=expected_volume
    )


def test_surface_rzfourier_from_nescoil_input_object_api_parity() -> None:
    plasma_surface = SurfaceRZFourier.from_nescoil_input(
        TEST_DIR / "nescin.LandremanPaul2021_QA", "plasma"
    )
    reference_surface = SurfaceRZFourier.from_vmec_input(
        TEST_DIR / "input.LandremanPaul2021_QA"
    )
    _assert_surface_parity(plasma_surface)
    np.testing.assert_allclose(
        plasma_surface.volume(),
        reference_surface.volume(),
        rtol=0.0,
        atol=2e-1,
    )


def test_surface_rzfourier_copy_object_api_parity() -> None:
    surface = SurfaceRZFourier(mpol=4, ntor=5, nfp=3)
    for copied_surface in _surface_copy_variants(surface):
        _assert_surface_parity(copied_surface)
