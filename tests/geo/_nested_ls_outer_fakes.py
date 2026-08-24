"""Fake collaborators the nested-LS outer children are driven against.

Both outer children reach physics only through injected collaborators, which
is what lets their real optimizer loop, real transaction and real payload
builder run under a test with no Boozer solve, no bundle on disk and no GPU.
The stand-ins for that world lived in two test modules with near-identical
bodies and had already drifted apart, so they live here once instead.

``J(c) = c.c`` is the fake outer objective, and it is the whole point of the
idiom: a published row claiming to be the objective can be checked against a
number this module computes rather than against the code that set the claim.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from simsopt_jax_adapters.geo.flat675.policy import FLAT675_OBJECTIVE_TERM_KEYS


def objective_at(coil_dofs: object) -> float:
    """The fake outer objective ``J(c) = c.c`` these tests measure rows against."""

    coils = np.asarray(coil_dofs, dtype=np.float64)
    return float(np.dot(coils, coils))


class _FakeSurface:
    def __init__(self, dofs: NDArray[np.float64]) -> None:
        self._dofs = np.array(dofs, dtype=np.float64, copy=True)

    def get_dofs(self) -> NDArray[np.float64]:
        return np.array(self._dofs, copy=True)

    def set_dofs(self, dofs: object) -> None:
        self._dofs = np.array(dofs, dtype=np.float64, copy=True)


class _FakeBiotSavart:
    def __init__(self, coil_dofs: NDArray[np.float64]) -> None:
        self.x = np.array(coil_dofs, dtype=np.float64, copy=True)


class _FakeBoozer:
    def __init__(
        self, coil_dofs: NDArray[np.float64], surface_dofs: NDArray[np.float64]
    ) -> None:
        self.biotsavart = _FakeBiotSavart(coil_dofs)
        self.surface = _FakeSurface(surface_dofs)
        self.need_to_run_code = False


class _FakeJaxBoozer(_FakeBoozer):
    """``_FakeBoozer`` plus the coil-data refresh the JAX restore path calls.

    ``refresh_count`` is the superset behaviour of the two copies this module
    replaced: it costs nothing to keep and it is what a restore-path test
    asserts on.
    """

    def __init__(
        self, coil_dofs: NDArray[np.float64], surface_dofs: NDArray[np.float64]
    ) -> None:
        super().__init__(coil_dofs, surface_dofs)
        self.refresh_count = 0

    def _refresh_coil_data(self) -> None:
        self.refresh_count += 1


class _FakeJaxOuterState:
    """``NestedLsOuterState``'s anchor and inner telemetry, nothing else.

    Every field starts at a value no real solve produces, so a lane that
    forgets to set one publishes an obvious sentinel rather than a plausible
    number.
    """

    def __init__(self) -> None:
        self.anchor_surface_dofs = np.array([-1.0], dtype=np.float64)
        self.anchor_iota = -1.0
        self.anchor_G = -1.0
        self.inner_iterations = -1
        self.inner_grad_l2 = -1.0
        self.adjoint_live_eta = -1.0

    def set_anchor(self, surface_dofs: object, iota: float, G: float) -> None:
        self.anchor_surface_dofs = np.array(surface_dofs, dtype=np.float64, copy=True)
        self.anchor_iota = float(iota)
        self.anchor_G = float(G)


class _FakeObjective:
    """The native twin's eight-term objective, replaced by ``J(c) = c.c``."""

    def __init__(self, boozer: _FakeBoozer) -> None:
        self.boozer = boozer

    def evaluate(self) -> tuple[float, dict[str, float], NDArray[np.float64]]:
        coils = np.asarray(self.boozer.biotsavart.x, dtype=np.float64)
        value = objective_at(coils)
        terms = {key: value for key in FLAT675_OBJECTIVE_TERM_KEYS}
        return value, terms, 2.0 * coils
