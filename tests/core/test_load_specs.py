from pathlib import Path

import numpy as np
import pytest

from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(Path(__file__).resolve().parents[2] / "src")

from simsopt._core.json import (  # noqa: E402
    load_specs,
    save_biot_savart_spec,
    save_surface_rz_fourier_spec,
)
from simsopt._core.optimizable import load  # noqa: E402
from simsopt.jax_core.field import grouped_biot_savart_B_from_spec  # noqa: E402
from simsopt.jax_core.specs import (  # noqa: E402
    BiotSavartSpec,
    GroupedCoilSetSpec,
    SurfaceRZFourierSpec,
)
from simsopt.jax_core.surface_rzfourier import (  # noqa: E402
    surface_rz_fourier_dofs_from_spec,
)


def _write_legacy_outputs(root: Path) -> tuple[Path, Path]:
    pytest.importorskip("simsoptpp")
    from simsopt.field.biotsavart import BiotSavart
    from simsopt.field.coil import Current, coils_via_symmetries
    from simsopt.geo.curvexyzfourier import CurveXYZFourier
    from simsopt.geo.surfacerzfourier import SurfaceRZFourier

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

    assert isinstance(load(coil_output), GroupedCoilSetSpec)
    assert isinstance(load(surface_output), SurfaceRZFourierSpec)
    assert isinstance(load_specs(coil_output)["coil_set_spec"], GroupedCoilSetSpec)
    assert isinstance(load_specs(surface_output)["surface_spec"], SurfaceRZFourierSpec)


def test_biot_savart_restart_spec_round_trips_with_dof_extraction(tmp_path):
    from simsopt.field.biotsavart_jax_backend import (  # noqa: E402
        BiotSavartJAX,
        SpecBackedBiotSavartJAX,
    )
    from simsopt.jax_core import make_biot_savart_spec  # noqa: E402

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


def test_load_reconstructs_legacy_cws_curve_artifact():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "3_Advanced"
        / "optimization_cws_singlestage_nfp2_QA_ncoils3_axiTorus"
        / "coils"
        / "biot_savart_opt_maxmode3.json"
    )
    if not path.exists():
        pytest.skip("legacy CWS example artifact is not present")

    legacy_bs = load(path)

    assert len(legacy_bs.coils) == 12
    curve = legacy_bs.coils[0].curve
    assert type(curve).__name__ == "CurveCWSFourier"
    np.testing.assert_allclose(curve.surf.get_dofs(), np.asarray([1.0, 0.55, 0.55]))
    np.testing.assert_allclose(
        curve.get_dofs()[:5],
        np.asarray(
            [
                1.0,
                -0.08893490083714978,
                0.026334506334377627,
                0.05488038834701336,
                -0.09845945463373941,
            ]
        ),
    )


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
