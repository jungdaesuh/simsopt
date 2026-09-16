import json
from pathlib import Path

import numpy as np
import pytest

from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(Path(__file__).resolve().parents[2] / "src")

pytest.importorskip("simsoptpp")

from simsopt._core.optimizable import load  # noqa: E402
from simsopt.field.biotsavart import BiotSavart  # noqa: E402
from simsopt.field.coil import Current, coils_via_symmetries  # noqa: E402
from simsopt.geo.curvexyzfourier import CurveXYZFourier  # noqa: E402
from simsopt.geo.surfacerzfourier import SurfaceRZFourier  # noqa: E402
from simsopt_jax.core import make_biot_savart_spec  # noqa: E402
from simsopt_jax.core.field import grouped_biot_savart_B_from_spec  # noqa: E402
from simsopt_jax.core.specs import (  # noqa: E402
    BiotSavartSpec,
    CoilGroupSpec,
    GroupedCoilSetSpec,
    SurfaceRZFourierSpec,
)
from simsopt_jax.core.surface_rzfourier import (  # noqa: E402
    surface_rz_fourier_dofs_from_spec,
)
from simsopt_jax_adapters.field.biotsavart_backend import (  # noqa: E402
    BiotSavartJAX,
    SpecBackedBiotSavartJAX,
)
from simsopt_jax_adapters.io.specs import (  # noqa: E402
    load_specs,
    save_biot_savart_spec,
    save_surface_rz_fourier_spec,
)


def _write_legacy_outputs(root: Path) -> tuple[Path, Path]:
    curve = CurveXYZFourier(8, 1)
    coeffs = curve.dofs_matrix
    coeffs[1][0] = 1.0
    coeffs[2][1] = 1.0
    curve.set_dofs(np.concatenate(coeffs))
    coils = coils_via_symmetries([curve], [Current(1.0)], nfp=2, stellsym=True)
    biot_savart = BiotSavart(coils)
    biot_savart.set_points(np.asarray([[0.2, 0.1, 0.3], [0.3, 0.2, 0.4]]))

    surface = SurfaceRZFourier(
        nfp=2,
        stellsym=True,
        mpol=1,
        ntor=1,
        quadpoints_phi=np.linspace(0, 1, 4, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 4, endpoint=False),
    )
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.2)
    surface.set_zs(1, 0, 0.2)

    biot_savart_path = root / "biot_savart_opt.json"
    surface_path = root / "surf_opt.json"
    biot_savart.save(filename=biot_savart_path)
    surface.save(filename=surface_path)
    return biot_savart_path, surface_path


def test_load_specs_reads_legacy_json_without_losing_geometry_state(tmp_path):
    biot_savart_path, surface_path = _write_legacy_outputs(tmp_path)

    coil_set_spec = load_specs(biot_savart_path)["coil_set_spec"]
    surface_spec = load_specs(surface_path)["surface_spec"]

    assert isinstance(coil_set_spec, GroupedCoilSetSpec)
    assert isinstance(surface_spec, SurfaceRZFourierSpec)

    legacy_bs = load(biot_savart_path)
    assert len(legacy_bs.coils) == 4
    assert any(type(coil.curve).__name__ == "RotatedCurve" for coil in legacy_bs.coils)
    assert any(
        type(coil.current).__name__ == "ScaledCurrent" for coil in legacy_bs.coils
    )
    points = legacy_bs.get_points_cart()
    np.testing.assert_allclose(
        np.asarray(grouped_biot_savart_B_from_spec(points, coil_set_spec)),
        np.asarray(legacy_bs.B()),
        rtol=1e-12,
        atol=1e-12,
    )

    legacy_surface = load(surface_path)
    np.testing.assert_array_equal(
        np.asarray(surface_rz_fourier_dofs_from_spec(surface_spec)),
        np.asarray(legacy_surface.get_dofs()),
    )


def test_spec_writers_round_trip_through_load_and_load_specs(tmp_path):
    biot_savart_path, surface_path = _write_legacy_outputs(tmp_path)
    coil_set_spec = load_specs(biot_savart_path)["coil_set_spec"]
    surface_spec = load_specs(surface_path)["surface_spec"]
    coil_output = tmp_path / "biot_savart_spec.json"
    surface_output = tmp_path / "surface_spec.json"

    save_biot_savart_spec(coil_output, coil_set_spec)
    save_surface_rz_fourier_spec(surface_output, surface_spec)

    assert isinstance(load_specs(coil_output)["coil_set_spec"], GroupedCoilSetSpec)
    assert isinstance(load_specs(surface_output)["surface_spec"], SurfaceRZFourierSpec)
    assert isinstance(load(coil_output), GroupedCoilSetSpec)
    assert isinstance(load(surface_output), SurfaceRZFourierSpec)


def test_biot_savart_restart_spec_round_trips_with_dof_extraction(tmp_path):
    biot_savart_path, _surface_path = _write_legacy_outputs(tmp_path)
    legacy_bs = load(biot_savart_path)
    legacy_bs_jax = BiotSavartJAX(legacy_bs.coils)
    legacy_points = np.asarray(legacy_bs.get_points_cart(), dtype=np.float64)
    restart_spec = make_biot_savart_spec(
        coil_dof_extraction=legacy_bs_jax.coil_dof_extraction_spec(),
        coil_dofs=legacy_bs_jax.x,
    )
    restart_output = tmp_path / "biot_savart_restart_spec.json"

    save_biot_savart_spec(restart_output, restart_spec)
    loaded = load_specs(restart_output)
    spec_backed_bs = SpecBackedBiotSavartJAX(loaded["biot_savart_spec"])
    spec_backed_bs.set_points(legacy_points)

    assert isinstance(load(restart_output), BiotSavartSpec)
    assert isinstance(loaded["biot_savart_spec"], BiotSavartSpec)
    assert isinstance(loaded["coil_set_spec"], GroupedCoilSetSpec)
    np.testing.assert_allclose(
        np.asarray(spec_backed_bs.B()),
        np.asarray(legacy_bs.B()),
        rtol=1e-12,
        atol=1e-12,
    )


def test_spec_backed_biot_savart_restart_dofs_are_mutable(tmp_path):
    biot_savart_path, _surface_path = _write_legacy_outputs(tmp_path)
    legacy_bs = load(biot_savart_path)
    legacy_bs_jax = BiotSavartJAX(legacy_bs.coils)
    restart_spec = make_biot_savart_spec(
        coil_dof_extraction=legacy_bs_jax.coil_dof_extraction_spec(),
        coil_dofs=legacy_bs_jax.x,
    )

    spec_backed_bs = SpecBackedBiotSavartJAX(restart_spec)
    spec_backed_bs.local_x = spec_backed_bs.local_x.copy()

    np.testing.assert_allclose(spec_backed_bs.local_x, legacy_bs_jax.x)


def test_biot_savart_spec_writer_uses_explicit_jax_host_boundary(tmp_path):
    jax = pytest.importorskip("jax")
    dtype = np.float64 if jax.config.jax_enable_x64 else np.float32
    group = CoilGroupSpec(
        gammas=jax.device_put(np.zeros((1, 2, 3), dtype=dtype)),
        gammadashs=jax.device_put(np.ones((1, 2, 3), dtype=dtype)),
        currents=jax.device_put(np.asarray([1.5], dtype=dtype)),
        coil_indices=(0,),
    )
    spec = GroupedCoilSetSpec(groups=(group,))
    output = tmp_path / "strict_biot_savart_spec.json"

    with jax.transfer_guard("disallow"):
        save_biot_savart_spec(output, spec)

    loaded = load_specs(output)["coil_set_spec"]
    assert isinstance(loaded, GroupedCoilSetSpec)
    np.testing.assert_allclose(
        np.asarray(loaded.groups[0].gammas),
        np.asarray(group.gammas),
    )
    np.testing.assert_allclose(
        np.asarray(loaded.groups[0].gammadashs),
        np.asarray(group.gammadashs),
    )
    np.testing.assert_allclose(
        np.asarray(loaded.groups[0].currents),
        np.asarray(group.currents),
    )


# Legacy CurveCWSFourier artifact, hand-built at order 1 in the GSON shape
# upstream simsopt 20265c3fa emitted for
# examples/3_Advanced/optimization_cws_singlestage_nfp2_QA_ncoils3_axiTorus/
# coils/biot_savart_opt_maxmode3.json.
# Layout of the legacy dof vector (upstream src/simsoptpp/curvecwsfourier.h,
# get_dofs/set_dofs_impl), angles in radians:
#   [theta_l, theta_c[0..order], theta_s[1..order],
#    phi_l,   phi_c[0..order],   phi_s[1..order]]
# Every harmonic below is a power-of-two multiple of pi, so dividing by 2 pi
# gives an exactly representable number of turns and the expectations further
# down are hand-written literals rather than a second copy of the conversion.
#   theta_l = 1         -> G = 1
#   theta_c[0] =  pi/2  ->  0.25   turns
#   theta_c[1] =  pi/4  ->  0.125  turns
#   theta_s[1] = -pi/8  -> -0.0625 turns
#   phi_l   = 2         -> H = 2
#   phi_c[0] =  pi      ->  0.5    turns
#   phi_c[1] = -pi/2    -> -0.25   turns
#   phi_s[1] =  pi/4    ->  0.125  turns
_LEGACY_CWS_ORDER = 1
_LEGACY_CWS_IDOFS = [1.0, 0.5, 0.25]  # rc(0,0), rc(1,0), zs(1,0)
_LEGACY_CWS_QUADPOINTS = [0.0, 0.25, 0.5, 0.75]
_LEGACY_CWS_X = [
    1.0,
    1.5707963267948966,
    0.7853981633974483,
    -0.39269908169872414,
    2.0,
    3.141592653589793,
    -1.5707963267948966,
    0.7853981633974483,
]
# Upstream fixes exactly the two secular dofs (indices 0 and 2 * order + 2).
_LEGACY_CWS_FREE = [False, True, True, True, False, True, True, True]
# This class' dof order is [phic(0), phic(1), phis(1), thetac(0), thetac(1),
# thetas(1)]; the values above in turns, read off by hand.
_EXPECTED_MODES = [0.5, -0.25, 0.125, 0.25, 0.125, -0.0625]
_EXPECTED_DOF_NAMES = [
    "phic(0)",
    "phic(1)",
    "phis(1)",
    "thetac(0)",
    "thetac(1)",
    "thetas(1)",
]
# Hand evaluation of the upstream angle map at the four quadpoints, in turns:
#   theta(t) / 2 pi = G * t + 0.25 + 0.125 cos(2 pi t) - 0.0625 sin(2 pi t)
#   phi(t)   / 2 pi = H * t + 0.5  - 0.25  cos(2 pi t) + 0.125  sin(2 pi t)
# with cos(2 pi t) = 1, 0, -1, 0 and sin(2 pi t) = 0, 1, 0, -1.
_EXPECTED_THETA_TURNS = [0.375, 0.4375, 0.625, 1.0625]
_EXPECTED_PHI_TURNS = [0.25, 1.125, 1.75, 1.875]


def _write_legacy_cws_artifact(path: Path, x=None, free=None) -> Path:
    """Write the order-1 legacy artifact in the GSON shape upstream emitted."""
    artifact = {
        "@module": "simsopt._core.json",
        "@class": "SIMSON",
        "@version": "0.0.post3377+g102b6e7",
        "graph": {"$type": "ref", "value": "CurveCWSFourier1"},
        "simsopt_objs": {
            "CurveCWSFourier1": {
                "@module": "simsopt.geo.curvecwsfourier",
                "@class": "CurveCWSFourier",
                "@name": "CurveCWSFourier1",
                "@version": "0.0.post3377+g102b6e7",
                "dofs": {"$type": "ref", "value": "6136106624"},
                "idofs": _LEGACY_CWS_IDOFS,
                "mpol": 1,
                "nfp": 2,
                "ntor": 0,
                "order": _LEGACY_CWS_ORDER,
                "quadpoints": {
                    "@module": "numpy",
                    "@class": "array",
                    "dtype": "float64",
                    "data": _LEGACY_CWS_QUADPOINTS,
                },
                "stellsym": True,
            },
            "6136106624": {
                "@module": "simsopt._core.optimizable",
                "@class": "DOFs",
                "@name": "6136106624",
                "@version": "0.0.post3377+g102b6e7",
                "x": {
                    "@module": "numpy",
                    "@class": "array",
                    "dtype": "float64",
                    "data": _LEGACY_CWS_X if x is None else x,
                },
                "free": {
                    "@module": "numpy",
                    "@class": "array",
                    "dtype": "bool",
                    "data": _LEGACY_CWS_FREE if free is None else free,
                },
            },
        },
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def _expected_gamma() -> np.ndarray:
    """Position of the hand-evaluated angles on the fixture's winding surface.

    The fixture surface is stellarator symmetric with ``mpol=1``, ``ntor=0``, so
    upstream ``curvecwsfourier.cpp::gamma_impl`` reduces to
    ``r = rc(0,0) + rc(1,0) cos(theta)``, ``z = zs(1,0) sin(theta)``,
    ``(x, y) = (r cos(phi), r sin(phi))`` -- the ``nfp * n * phi`` term drops out
    because ``n = 0``. Only the hand-derived angles enter, not the dof layout.
    """
    rc00, rc10, zs10 = _LEGACY_CWS_IDOFS
    theta = 2.0 * np.pi * np.asarray(_EXPECTED_THETA_TURNS)
    phi = 2.0 * np.pi * np.asarray(_EXPECTED_PHI_TURNS)
    radius = rc00 + rc10 * np.cos(theta)
    return np.column_stack(
        (radius * np.cos(phi), radius * np.sin(phi), zs10 * np.sin(theta))
    )


def test_load_reconstructs_legacy_cws_curve_artifact(tmp_path):
    curve = load(_write_legacy_cws_artifact(tmp_path / "legacy_cws.json"))

    assert type(curve).__name__ == "CurveCWSFourier"
    assert curve.order == _LEGACY_CWS_ORDER
    assert (curve.surf.nfp, curve.surf.mpol, curve.surf.ntor) == (2, 1, 0)
    assert curve.surf.stellsym
    np.testing.assert_array_equal(curve.surf.get_dofs(), np.asarray(_LEGACY_CWS_IDOFS))
    np.testing.assert_array_equal(curve.quadpoints, np.asarray(_LEGACY_CWS_QUADPOINTS))
    # The legacy secular dofs theta_l/phi_l are this class' winding numbers.
    assert (curve.G, curve.H) == (1, 2)
    # The remaining legacy dofs are the same harmonics, reordered from the
    # (theta, phi) blocks to this class' (phi, theta) blocks and rescaled from
    # radians to turns. Both expectations are exact literals: a block swap
    # permutes them and a missing 1 / (2 pi) rescales them.
    assert list(curve.local_full_dof_names) == _EXPECTED_DOF_NAMES
    np.testing.assert_array_equal(
        curve.local_full_x, np.asarray(_EXPECTED_MODES, dtype=np.float64)
    )
    # The secular dofs were fixed upstream and became G/H, so every surviving
    # dof of this fixture is free.
    assert curve.local_dofs_free_status.all()

    gamma_2d = np.asarray(curve.gamma_2d())
    np.testing.assert_allclose(
        gamma_2d[:, 0], np.asarray(_EXPECTED_PHI_TURNS), rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        gamma_2d[:, 1], np.asarray(_EXPECTED_THETA_TURNS), rtol=0.0, atol=1e-15
    )
    gamma = np.asarray(curve.gamma())
    assert np.isfinite(gamma).all()
    np.testing.assert_allclose(gamma, _expected_gamma(), rtol=0.0, atol=1e-14)

    shipped = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "3_Advanced"
        / "optimization_cws_singlestage_nfp2_QA_ncoils3_axiTorus"
        / "coils"
        / "biot_savart_opt_maxmode3.json"
    )
    if shipped.exists():
        legacy_bs = load(shipped)
        assert len(legacy_bs.coils) == 12
        shipped_curve = legacy_bs.coils[0].curve
        assert type(shipped_curve).__name__ == "CurveCWSFourier"
        assert (shipped_curve.order, shipped_curve.G, shipped_curve.H) == (16, 1, 0)
        np.testing.assert_allclose(
            shipped_curve.surf.get_dofs(), np.asarray([1.0, 0.55, 0.55])
        )
        np.testing.assert_allclose(
            shipped_curve.get_dofs()[:5],
            np.asarray(
                [
                    0.29640353431707916,
                    -0.13392743935529472,
                    0.23424750931990623,
                    -0.15823960909088475,
                    0.14538297028868877,
                ]
            )
            / (2.0 * np.pi),
        )
        # Upstream fixes only the two secular dofs, so all 66 harmonics are free.
        assert shipped_curve.local_dofs_free_status.all()


def test_legacy_cws_curve_keeps_the_legacy_free_mask(tmp_path):
    # Fix theta_s[1] (legacy index 3), which maps to this class' thetas(1).
    free = list(_LEGACY_CWS_FREE)
    free[3] = False
    curve = load(
        _write_legacy_cws_artifact(tmp_path / "legacy_cws_fixed.json", free=free)
    )

    np.testing.assert_array_equal(
        curve.local_dofs_free_status,
        np.asarray([True, True, True, True, True, False]),
    )
    assert list(curve.local_dof_names) == _EXPECTED_DOF_NAMES[:5]
    # Fixing a dof must not disturb the values.
    np.testing.assert_array_equal(
        curve.local_full_x, np.asarray(_EXPECTED_MODES, dtype=np.float64)
    )


def test_legacy_cws_curve_rejects_a_free_secular_dof(tmp_path):
    free = list(_LEGACY_CWS_FREE)
    free[2 * _LEGACY_CWS_ORDER + 2] = True
    path = _write_legacy_cws_artifact(
        tmp_path / "legacy_cws_free_secular.json", free=free
    )

    with pytest.raises(ValueError, match="are free dofs"):
        load(path)


def test_legacy_cws_curve_rejects_a_non_finite_secular_dof(tmp_path):
    x = list(_LEGACY_CWS_X)
    x[2 * _LEGACY_CWS_ORDER + 2] = float("nan")
    path = _write_legacy_cws_artifact(tmp_path / "legacy_cws_nan_secular.json", x=x)

    with pytest.raises(ValueError, match="are not finite"):
        load(path)


def test_legacy_cws_curve_rejects_a_fractional_secular_dof(tmp_path):
    x = list(_LEGACY_CWS_X)
    x[0] = 1.5
    path = _write_legacy_cws_artifact(tmp_path / "legacy_cws_fractional.json", x=x)

    with pytest.raises(ValueError, match="are not whole turns"):
        load(path)


def test_load_specs_rejects_non_simson_wrapper_module(tmp_path):
    path = tmp_path / "unsupported_wrapper.json"
    path.write_text(
        """{
  "@module": "simsopt.unsupported",
  "@class": "SIMSON",
  "graph": {"$type": "ref", "value": "Unsupported1"},
  "simsopt_objs": {}
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected GSON SIMSON wrapper"):
        load_specs(path)


def test_load_specs_rejects_unsupported_gson_value_type(tmp_path):
    path = tmp_path / "unsupported_value_type.json"
    path.write_text(
        """{
  "@module": "simsopt._core.json",
  "@class": "SIMSON",
  "graph": {"$type": "inline", "value": "Unsupported1"},
  "simsopt_objs": {}
}
""",
        encoding="utf-8",
    )

    with pytest.raises(NotImplementedError, match="Unsupported GSON value type"):
        load_specs(path)


def test_load_specs_rejects_unsupported_gson_class(tmp_path):
    path = tmp_path / "unsupported.json"
    path.write_text(
        """{
  "@module": "simsopt._core.json",
  "@class": "SIMSON",
  "graph": {"$type": "ref", "value": "Unsupported1"},
  "simsopt_objs": {
    "Unsupported1": {
      "@module": "simsopt.unsupported",
      "@class": "Unsupported"
    }
  }
}
""",
        encoding="utf-8",
    )

    with pytest.raises(NotImplementedError, match="simsopt.unsupported.Unsupported"):
        load_specs(path)
