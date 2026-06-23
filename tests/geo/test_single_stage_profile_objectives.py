"""Tests for the per-surface iota- and volume-PROFILE single-stage objectives.

Both builders compose a per-surface ``QuadraticPenalty`` against a profile target
and weighted-average them via ``average_surface_objectives``; the coil-DOF gradient
rides the proven ``Iotas`` (iota) / ``VolumeBoozer`` (volume) Boozer adjoints, so no
new gradient code exists -- these tests prove the COMPOSITION is correct, non-vacuous,
and default-OFF inert.

Two complementary gradient proofs per builder:
  * a FAST controllable-stand-in test (no BoozerSurface solve) whose stand-in J is
    LINEAR in x -- this makes the composed penalty actually depend on x (a constant-J
    stand-in would give a zero gradient and a vacuous test). Because the composite is
    then exactly quadratic, the central difference is EXACT, so we assert it equals the
    analytic directional derivative to tight atol rather than a Taylor-convergence ratio.
  * a REAL ``get_boozer_surface`` Taylor test on TWO independent surface stacks, where
    ``Iotas``/``VolumeBoozer`` are nonlinear in x, giving genuine Taylor convergence --
    the doc's mandated "ride the proven adjoint on a >=2-surface stack" proof. This also
    pins that ``obj.x`` is the UNION of both stacks' free coil dofs.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

from simsopt._core.derivative import Derivative, derivative_dec
from simsopt._core.optimizable import Optimizable
from simsopt.geo.surfaceobjectives import Iotas, Volume, VolumeBoozer

EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from banana_opt.single_stage_objectives import (  # noqa: E402
    build_single_stage_iota_profile_objective,
    build_single_stage_volume_profile_objective,
)

from .surface_test_helpers import get_boozer_surface  # noqa: E402


def taylor_test1(f, df, x, epsilons=None, direction=None, atol=1e-9):
    """Copied verbatim from test_surface_objectives.py:19 (proven Taylor harness)."""
    np.random.seed(1)
    f(x)
    if direction is None:
        direction = np.random.rand(*(x.shape)) - 0.5
    dfx = df(x) @ direction
    if epsilons is None:
        epsilons = np.power(2.0, -np.asarray(range(10, 20)))
    err_old = 1e9
    for eps in epsilons:
        fpluseps = f(x + eps * direction)
        fminuseps = f(x - eps * direction)
        dfest = (fpluseps - fminuseps) / (2 * eps)
        err = np.linalg.norm(dfest - dfx)
        np.testing.assert_array_less(err, max(atol, 0.31 * err_old))
        err_old = err


class _LinearLeafTerm(Optimizable):
    """Real Optimizable standing in for Iotas/VolumeBoozer, LINEAR in its own dofs.

    ``J() = base + d . x`` and ``dJ()`` is a genuine ``Derivative`` keyed on this term,
    so the chain rule flows exactly as the true Boozer adjoint would. Unlike a constant
    stand-in, J depends on x, so the composed profile penalty has a NON-ZERO gradient
    (a wrong/zero gradient fails the exact-match assertion below). Distinct ``d`` per
    instance gives each surface its own dofs, so the composite ``.x`` is their union.
    """

    def __init__(self, base, d):
        self._base = float(base)
        self._d = np.asarray(d, dtype=float)
        Optimizable.__init__(self, x0=np.zeros(self._d.size))

    def J(self):
        return self._base + float(self._d @ self.x)

    @derivative_dec
    def dJ(self):
        return Derivative({self: self._d.copy()})

    return_fn_map = {"J": J, "dJ": dJ}


def _assert_exact_directional_gradient(obj, atol=1e-10):
    """Central diff == analytic dJ.direction to ~machine eps (composite is quadratic).

    Also asserts the gradient is non-vacuous (nonzero norm), so a dead/zeroed gradient
    cannot pass. Returns the analytic gradient for the caller.
    """
    x0 = obj.x.copy()
    np.random.seed(1)
    direction = np.random.rand(*x0.shape) - 0.5

    def f(x):
        obj.x = x
        return obj.J()

    def df(x):
        obj.x = x
        return obj.dJ()

    analytic = df(x0) @ direction
    eps = 1e-6
    central = (f(x0 + eps * direction) - f(x0 - eps * direction)) / (2.0 * eps)
    np.testing.assert_allclose(central, analytic, rtol=1e-6, atol=atol)
    grad = df(x0)
    assert np.linalg.norm(grad) > 0.0, "profile objective gradient must be non-vacuous"
    return grad


# --------------------------------------------------------------------------------------
# Iota-profile builder
# --------------------------------------------------------------------------------------
class IotaProfileBuilderFastTests(unittest.TestCase):
    def test_x_is_union_of_surface_dofs(self):
        t0 = _LinearLeafTerm(0.20, [0.5, -0.25, 0.75])
        t1 = _LinearLeafTerm(0.10, [-0.3, 0.6])
        obj = build_single_stage_iota_profile_objective(
            [t0, t1], surface_labels=[0.5, 1.0],
            iota_edge_target=0.09, iota_slope_target=-0.05,
        )
        self.assertIsNotNone(obj)
        self.assertEqual(obj.x.size, t0.x.size + t1.x.size)  # union: 3 + 2 = 5

    def test_gradient_matches_central_difference_and_is_nonzero(self):
        # Targets deliberately != realized iotas so the two-sided identity penalty is
        # active on every surface (nonzero gradient).
        t0 = _LinearLeafTerm(0.20, [0.5, -0.25, 0.75])
        t1 = _LinearLeafTerm(0.10, [-0.3, 0.6])
        obj = build_single_stage_iota_profile_objective(
            [t0, t1], surface_labels=[0.5, 1.0],
            iota_edge_target=0.09, iota_slope_target=-0.05,
        )
        _assert_exact_directional_gradient(obj)

    def test_none_for_single_surface(self):
        t0 = _LinearLeafTerm(0.2, [1.0])
        self.assertIsNone(
            build_single_stage_iota_profile_objective(
                [t0], surface_labels=[1.0],
                iota_edge_target=0.1, iota_slope_target=0.0,
            )
        )

    def test_length_mismatch_raises(self):
        t0 = _LinearLeafTerm(0.2, [1.0])
        t1 = _LinearLeafTerm(0.1, [1.0])
        with self.assertRaises(ValueError):
            build_single_stage_iota_profile_objective(
                [t0, t1], surface_labels=[1.0],  # 2 terms, 1 label
                iota_edge_target=0.1, iota_slope_target=0.0,
            )


class IotaProfileBuilderRealTaylorTests(unittest.TestCase):
    def test_two_boozer_stacks_taylor(self):
        # Two INDEPENDENT Boozer stacks (fresh coils each) -> obj.x is their union;
        # Iotas is nonlinear in x -> genuine Taylor convergence (rides the proven
        # Iotas adjoint, test_surface_objectives.py).
        bs0, bsurf0 = get_boozer_surface(
            label="Volume", boozer_type="ls", optimize_G=True, weight_inv_modB=False
        )
        bs1, bsurf1 = get_boozer_surface(
            label="Volume", boozer_type="ls", optimize_G=True, weight_inv_modB=False
        )
        terms = [Iotas(bsurf0), Iotas(bsurf1)]
        # Slope 0 + edge target near the converged iota keeps the penalty well-scaled;
        # gradient is still nonzero because targets != exact realized iotas.
        obj = build_single_stage_iota_profile_objective(
            terms, surface_labels=[0.5, 1.0],
            iota_edge_target=-0.40, iota_slope_target=0.0,
        )
        self.assertIsNotNone(obj)
        self.assertEqual(obj.x.size, bs0.x.size + bs1.x.size)
        coeffs = obj.x.copy()

        def f(dofs):
            obj.x = dofs
            return obj.J()

        def df(dofs):
            obj.x = dofs
            return obj.dJ()

        taylor_test1(
            f, df, coeffs,
            epsilons=np.power(2.0, -np.asarray(range(13, 19))), atol=2e-8,
        )


# --------------------------------------------------------------------------------------
# Volume-profile builder
# --------------------------------------------------------------------------------------
class VolumeProfileBuilderFastTests(unittest.TestCase):
    def test_x_is_union_and_gradient_nonzero(self):
        v0 = _LinearLeafTerm(1.00, [0.4, -0.6, 0.2])
        v1 = _LinearLeafTerm(1.20, [0.7, -0.1])
        obj = build_single_stage_volume_profile_objective([v0, v1], [0.98, 1.25])
        self.assertIsNotNone(obj)
        self.assertEqual(obj.x.size, v0.x.size + v1.x.size)
        _assert_exact_directional_gradient(obj)

    def test_none_for_single_surface(self):
        v0 = _LinearLeafTerm(1.0, [1.0])
        self.assertIsNone(build_single_stage_volume_profile_objective([v0], [1.0]))

    def test_length_mismatch_raises(self):
        v0 = _LinearLeafTerm(1.0, [1.0])
        v1 = _LinearLeafTerm(1.2, [1.0])
        with self.assertRaises(ValueError):
            build_single_stage_volume_profile_objective([v0, v1], [1.0])


class VolumeProfileBuilderRealTaylorTests(unittest.TestCase):
    def test_two_boozer_stacks_taylor_volumeboozer(self):
        bs0, bsurf0 = get_boozer_surface(
            label="Volume", boozer_type="ls", optimize_G=True, weight_inv_modB=False
        )
        bs1, bsurf1 = get_boozer_surface(
            label="Volume", boozer_type="ls", optimize_G=True, weight_inv_modB=False
        )
        v0, v1 = VolumeBoozer(bsurf0), VolumeBoozer(bsurf1)
        targets = [float(v0.J()) * 0.98, float(v1.J()) * 1.02]  # != realized -> active
        obj = build_single_stage_volume_profile_objective([v0, v1], targets)
        self.assertIsNotNone(obj)
        self.assertEqual(obj.x.size, bs0.x.size + bs1.x.size)
        coeffs = obj.x.copy()

        def f(dofs):
            obj.x = dofs
            return obj.J()

        def df(dofs):
            obj.x = dofs
            return obj.dJ()

        taylor_test1(
            f, df, coeffs,
            epsilons=np.power(2.0, -np.asarray(range(13, 18))), atol=1e-8,
        )

    def test_bare_volume_term_is_coil_blind_dead_gradient(self):
        # The trap the docstring warns of: bare Volume(surface) has NO coil ancestor, so
        # feeding it to the volume-profile builder cannot move the coils. Mirror
        # test_surface_objectives.py::test_bare_volume_coil_gradient_is_dead and assert
        # ANCESTRY (robust), not .x-size (which counts free SURFACE dofs in this
        # standalone fixture and is nonzero).
        bs0, bsurf0 = get_boozer_surface(
            label="Volume", boozer_type="ls", optimize_G=True, weight_inv_modB=False
        )
        bs1, bsurf1 = get_boozer_surface(
            label="Volume", boozer_type="ls", optimize_G=True, weight_inv_modB=False
        )
        live = build_single_stage_volume_profile_objective(
            [VolumeBoozer(bsurf0), VolumeBoozer(bsurf1)], [1.0, 1.0]
        )
        bare = build_single_stage_volume_profile_objective(
            [Volume(bsurf0.surface), Volume(bsurf1.surface)], [1.0, 1.0]
        )
        self.assertTrue(any(a is bs0 for a in live.ancestors))
        self.assertTrue(any(a is bs1 for a in live.ancestors))
        self.assertFalse(any(a is bs0 for a in bare.ancestors))
        self.assertFalse(any(a is bs1 for a in bare.ancestors))


if __name__ == "__main__":
    unittest.main()
