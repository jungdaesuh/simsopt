"""``CurveCWSFourier`` must survive ``simsopt.save``/``simsopt.load`` unchanged.

The nested-endpoint native lane ships the flat-675 bridge's reconstructed coils
(``RotatedCurve(CurveCWSFourier)`` carried by a ``ScaledCurrent``) to a child
process through ``simsopt.save``/``simsopt.load``. That transport is only an
identity if the curve's serialization pair is an exact inverse, so these tests
compare the reloaded objects bitwise rather than approximately.
"""

import numpy as np
import pytest
from simsopt import load, save
from simsopt.field import BiotSavart
from simsopt.field.coil import Coil, Current, ScaledCurrent
from simsopt.geo import RotatedCurve, SurfaceRZFourier

from simsopt_jax_adapters.geo.curvecwsfourier import CurveCWSFourier

_FIELD_POINTS = np.array(
    [
        [1.45, 0.07, 0.11],
        [0.92, -0.63, -0.18],
        [-1.21, 0.44, 0.29],
    ],
    dtype=np.float64,
)


def _winding_surface() -> SurfaceRZFourier:
    """A winding surface built the way ``nested_bridge._native_curve`` builds one."""
    surface = SurfaceRZFourier(
        nfp=3,
        stellsym=True,
        mpol=1,
        ntor=1,
        quadpoints_phi=np.linspace(0.0, 1.0, 12, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 10, endpoint=False),
    )
    surface.set_dofs(
        np.array(
            [1.4, 0.07, 0.03, 0.62, -0.04, 0.05, 0.02, 0.58, -0.03],
            dtype=np.float64,
        )
    )
    return surface


def _bridge_curve() -> CurveCWSFourier:
    """A curve built the way ``nested_bridge._native_curve`` builds one."""
    curve = CurveCWSFourier(
        quadpoints=np.linspace(0.0, 1.0, 24, endpoint=False),
        order=2,
        surf=_winding_surface(),
        G=1.0,
        H=0.0,
    )
    curve.local_full_x = np.array(
        [0.03, 0.11, -0.05, 0.02, -0.01, 0.4, 0.07, -0.02, 0.06, 0.01],
        dtype=np.float64,
    )
    return curve


def test_curve_round_trip_is_bitwise(tmp_path):
    curve = _bridge_curve()
    path = tmp_path / "curve.json"
    save([curve], str(path))
    (reloaded,) = load(str(path))

    assert isinstance(reloaded, CurveCWSFourier)
    np.testing.assert_array_equal(reloaded.x, curve.x)
    np.testing.assert_array_equal(reloaded.local_full_x, curve.local_full_x)
    np.testing.assert_array_equal(reloaded.quadpoints, curve.quadpoints)
    assert reloaded.order == curve.order
    assert (reloaded.G, reloaded.H) == (curve.G, curve.H)
    np.testing.assert_array_equal(reloaded.gamma(), curve.gamma())


def test_winding_surface_round_trips_with_the_curve(tmp_path):
    curve = _bridge_curve()
    path = tmp_path / "curve.json"
    save([curve], str(path))
    (reloaded,) = load(str(path))

    surface = reloaded.surf
    assert isinstance(surface, SurfaceRZFourier)
    assert surface.nfp == curve.surf.nfp
    assert surface.stellsym == curve.surf.stellsym
    assert (surface.mpol, surface.ntor) == (curve.surf.mpol, curve.surf.ntor)
    np.testing.assert_array_equal(surface.get_dofs(), curve.surf.get_dofs())
    np.testing.assert_array_equal(surface.quadpoints_phi, curve.surf.quadpoints_phi)
    np.testing.assert_array_equal(surface.quadpoints_theta, curve.surf.quadpoints_theta)


def test_rotated_scaled_coil_field_round_trips_bitwise(tmp_path):
    base = _bridge_curve()
    rotated = RotatedCurve(base, 2.0 * np.pi / 3.0, True)
    coil = Coil(rotated, ScaledCurrent(Current(1.7e5), -1.0))
    field = BiotSavart([coil])
    field.set_points(_FIELD_POINTS)
    expected = field.B().copy()

    path = tmp_path / "field.json"
    save([field], str(path))
    (reloaded,) = load(str(path))

    (reloaded_coil,) = reloaded.coils
    assert isinstance(reloaded_coil.curve, RotatedCurve)
    assert isinstance(reloaded_coil.curve.curve, CurveCWSFourier)
    assert isinstance(reloaded_coil.current, ScaledCurrent)
    np.testing.assert_array_equal(reloaded_coil.curve.gamma(), rotated.gamma())
    assert reloaded_coil.current.get_value() == coil.current.get_value()

    reloaded.set_points(_FIELD_POINTS)
    np.testing.assert_array_equal(reloaded.B(), expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
