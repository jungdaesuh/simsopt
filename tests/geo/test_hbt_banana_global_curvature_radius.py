import sys
import types
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest


HBT_DIR = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "HBT_BANANA"
)
sys.path.insert(0, str(HBT_DIR / "new_objectives"))

# ``global_curvature_radius`` imports jax and simsopt at module scope, but its
# NumPy diagnostic needs neither. Where those packages are absent, stand in for
# them just long enough to import the module, then withdraw the stand-ins so
# the tests that genuinely need jax/simsopt still skip instead of running
# against fakes.


def _identity(obj):
    return obj


def _import_stubs():
    jax = types.ModuleType("jax")
    jax.grad = _identity
    jax.numpy = np
    derivative = types.ModuleType("simsopt._core.derivative")
    derivative.derivative_dec = _identity
    core = types.ModuleType("simsopt._core")
    core.Optimizable = object
    core.derivative = derivative
    geo_jit = types.ModuleType("simsopt.geo.jit")
    geo_jit.jit = _identity
    geo = types.ModuleType("simsopt.geo")
    geo.jit = geo_jit
    simsopt = types.ModuleType("simsopt")
    simsopt._core = core
    simsopt.geo = geo
    return {"jax": jax, "jax.numpy": np, "simsopt": simsopt,
            "simsopt._core": core, "simsopt._core.derivative": derivative,
            "simsopt.geo": geo, "simsopt.geo.jit": geo_jit}


_INSTALLED_STUBS = ()
if find_spec("jax") is None or find_spec("simsopt") is None:
    _stubs = {name: module for name, module in _import_stubs().items()
              if name not in sys.modules}
    sys.modules.update(_stubs)
    _INSTALLED_STUBS = tuple(_stubs)

from global_curvature_radius import (  # noqa: E402
    GlobalRadiusCurvature,
    global_curvature_radii,
)

for _name in _INSTALLED_STUBS:
    del sys.modules[_name]


UNDEFINED_RADIUS = 1.0e12


def _sampled_circle(radius, n_quadpoints, z=0.0):
    """Points and tangents of a planar circle of the given radius."""
    s = np.linspace(0.0, 2.0 * np.pi, n_quadpoints, endpoint=False)
    gamma = np.stack(
        [radius * np.cos(s), radius * np.sin(s), np.full_like(s, z)], axis=-1)
    gammadash = np.stack(
        [-radius * np.sin(s), radius * np.cos(s), np.zeros_like(s)], axis=-1)
    return gamma, gammadash


def _two_coaxial_lobes(separation, radius=1.0, n_quadpoints=128):
    """Two parallel circles, one at z = +separation/2, one at -separation/2."""
    upper = _sampled_circle(radius, n_quadpoints, z=0.5 * separation)
    lower = _sampled_circle(radius, n_quadpoints, z=-0.5 * separation)
    return (np.concatenate([upper[0], lower[0]]),
            np.concatenate([upper[1], lower[1]]))


# ── Pure-NumPy diagnostic ──────────────────────────────────────────────────

def test_circle_global_radius_equals_the_diameter_at_every_quadpoint():
    radius = 0.35

    radii = global_curvature_radii(*_sampled_circle(radius, 128))

    # S_C = |chord| / sin^2(half-angle) = 2R / sin(half-angle) on a circle, so
    # the minimum sits at the diametrically opposite quadpoint, where the
    # chord is perpendicular to the tangent and S_C collapses to the diameter.
    assert np.allclose(radii, 2.0 * radius, rtol=1e-12), (
        "a circle of radius R must report a global curvature radius of 2R at "
        f"every quadpoint; got min {radii.min()}, max {radii.max()}, "
        f"expected {2.0 * radius}")


def test_refining_the_quadpoint_grid_does_not_shrink_the_shortest_radius():
    radius = 0.35

    coarse = global_curvature_radii(*_sampled_circle(radius, 64)).min()
    fine = global_curvature_radii(*_sampled_circle(radius, 256)).min()

    # Neighbouring quadpoints crowd together under refinement, so a plain
    # pairwise-distance metric would collapse toward zero. The 1 - cos^2 factor
    # is what keeps adjacent pairs out of the minimum without an index mask.
    assert np.isclose(coarse, fine, rtol=1e-12), (
        "quadpoint refinement must not change the shortest radius of a fixed "
        f"circle; 64 points gave {coarse}, 256 points gave {fine}")
    assert np.isclose(fine, 2.0 * radius, rtol=1e-12), (
        "the refined shortest radius must remain the diameter, not an "
        f"adjacent-quadpoint artifact; got {fine}, expected {2.0 * radius}")


def test_moving_two_lobes_closer_lowers_the_shortest_radius():
    separations = [1.0, 0.5, 0.25, 0.1]

    shortest = [
        global_curvature_radii(*_two_coaxial_lobes(sep)).min()
        for sep in separations
    ]

    assert all(a > b for a, b in zip(shortest, shortest[1:])), (
        "the shortest global curvature radius must fall monotonically as the "
        f"two lobes approach each other; separations {separations} gave "
        f"{shortest}")
    assert np.allclose(shortest, separations, rtol=1e-9), (
        "for lobes facing each other across a perpendicular gap the shortest "
        f"radius is the gap itself; separations {separations} gave {shortest}")


def test_straight_segment_reports_the_undefined_sentinel_not_zero():
    t = np.linspace(0.0, 1.0, 16)
    gamma = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=-1)
    gammadash = np.tile(np.array([1.0, 0.0, 0.0]), (16, 1))

    radii = global_curvature_radii(gamma, gammadash)

    # Every chord is parallel to the tangent here, so no pair defines a
    # contact radius. The i == j self-pair has distance 0 and would drive the
    # minimum to 0 if it were not excluded from the row minimum.
    assert np.all(radii == UNDEFINED_RADIUS), (
        "a straight segment has no defined self-contact radius and must "
        f"report the {UNDEFINED_RADIUS} sentinel, never 0 from the excluded "
        f"self-pair; got min {radii.min()}, max {radii.max()}")


def test_off_diagonal_coincidence_reports_zero_radius_instead_of_sentinel():
    gamma = np.zeros((2, 3))
    gammadash = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    radii = global_curvature_radii(gamma, gammadash)

    assert np.all(radii == 0.0), (
        "distinct coincident quadpoints are a sampled self-intersection and "
        f"must report zero radius, not the undefined {UNDEFINED_RADIUS} "
        f"sentinel; got {radii}")


def test_radii_ignore_the_magnitude_of_gammadash():
    gamma, gammadash = _sampled_circle(0.35, 96)

    radii = global_curvature_radii(gamma, gammadash)
    rescaled = global_curvature_radii(gamma, 7.0 * gammadash)

    assert np.allclose(radii, rescaled, rtol=1e-12), (
        "the diagnostic uses the unit tangent, so reparameterising the curve "
        "at a different speed must not change the radii; got max deviation "
        f"{np.max(np.abs(radii - rescaled))}")


def test_radii_are_invariant_under_rotation_and_translation():
    gamma, gammadash = _two_coaxial_lobes(0.4)
    rotation, _ = np.linalg.qr(np.random.default_rng(7).standard_normal((3, 3)))
    shift = np.array([0.9, -0.4, 2.3])

    radii = global_curvature_radii(gamma, gammadash)
    moved = global_curvature_radii(gamma @ rotation.T + shift,
                                   gammadash @ rotation.T)

    assert np.allclose(radii, moved, rtol=1e-10), (
        "the global curvature radius is a rigid-motion invariant of the "
        "curve; rotating and translating changed it by up to "
        f"{np.max(np.abs(radii - moved))}")


# ── Optimizable wiring (needs jax and simsopt) ─────────────────────────────

def _circle_curve(simsopt_geo, radius=0.35, n_quadpoints=64):
    curve = simsopt_geo.CurveXYZFourier(n_quadpoints, 1)
    curve.set('xc(1)', radius)
    curve.set('ys(1)', radius)
    return curve


def _require_jax_x64():
    jax = pytest.importorskip("jax")
    if not jax.config.read("jax_enable_x64"):
        pytest.skip("the finite-difference gradient check requires JAX FP64")
    return jax


def test_objective_diagnostic_matches_the_module_level_function():
    pytest.importorskip("jax")
    simsopt_geo = pytest.importorskip("simsopt.geo")
    radius = 0.35
    curve = _circle_curve(simsopt_geo, radius)

    objective = GlobalRadiusCurvature(curve, minimum_radius=0.1)

    assert np.allclose(
        objective.global_curvature_radii(),
        global_curvature_radii(curve.gamma(), curve.gammadash())), (
        "the method must report exactly what the module-level diagnostic "
        "reports for the same curve arrays")
    assert np.isclose(objective.shortest_radius(), 2.0 * radius, rtol=1e-10), (
        "shortest_radius on a circle of radius R must be the diameter 2R; got "
        f"{objective.shortest_radius()}")


def test_objective_stores_its_thresholds_for_driver_logging():
    pytest.importorskip("jax")
    simsopt_geo = pytest.importorskip("simsopt.geo")
    curve = _circle_curve(simsopt_geo)

    objective = GlobalRadiusCurvature(curve, minimum_radius=0.12,
                                      exp_weight=0.02)

    assert (objective.minimum_radius, objective.exp_weight) == (0.12, 0.02), (
        "drivers log the active barrier thresholds off the objective, so both "
        "constructor scalars must survive as attributes; got "
        f"{(objective.minimum_radius, objective.exp_weight)}")


def test_barrier_grows_as_the_threshold_approaches_the_shortest_radius():
    pytest.importorskip("jax")
    simsopt_geo = pytest.importorskip("simsopt.geo")
    radius = 0.35
    curve = _circle_curve(simsopt_geo, radius)
    shortest = 2.0 * radius

    slack = GlobalRadiusCurvature(curve, minimum_radius=0.5 * shortest).J()
    tight = GlobalRadiusCurvature(curve, minimum_radius=0.99 * shortest).J()

    assert np.isfinite(slack) and slack > 0.0, (
        f"the barrier must stay finite and positive well below contact; got {slack}")
    assert tight > slack, (
        "raising the activation threshold toward the actual shortest radius "
        f"must raise the barrier; got {tight} at 0.99*R vs {slack} at 0.5*R")


def test_barrier_gradient_matches_central_differences():
    _require_jax_x64()
    simsopt_geo = pytest.importorskip("simsopt.geo")
    rng = np.random.default_rng(3)
    curve = _circle_curve(simsopt_geo, 0.35)
    curve.x = curve.x + 0.01 * rng.standard_normal(curve.x.size)
    objective = GlobalRadiusCurvature(curve, minimum_radius=0.69)

    direction = rng.standard_normal(curve.x.size)
    analytic = np.dot(objective.dJ(), direction)
    x0 = curve.x.copy()
    step = 1.0e-6
    curve.x = x0 + step * direction
    plus = objective.J()
    curve.x = x0 - step * direction
    minus = objective.J()
    curve.x = x0
    finite_difference = (plus - minus) / (2.0 * step)

    assert np.isclose(analytic, finite_difference, rtol=1.0e-5), (
        "the jax gradient must agree with a central difference of J along a "
        f"random dof direction; analytic {analytic}, finite difference "
        f"{finite_difference}")
