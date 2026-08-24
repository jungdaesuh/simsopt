"""Fake collaborators the nested-LS outer children are driven against.

Both outer children reach physics only through injected collaborators, which
is what lets their real optimizer loop, real transaction and real payload
builder run under a test with no Boozer solve, no bundle on disk and no GPU.
The stand-ins for that world lived in two test modules with near-identical
bodies and had already drifted apart, so they live here once instead.

``J(c) = c.c`` is the fake outer objective, and it is the whole point of the
idiom: a published row claiming to be the objective can be checked against a
number this module computes rather than against the code that set the claim.

A fake is only as good as the bugs it can express. This one was rewritten
after three independent adversarial reviews proved, by executing mutants,
that it could express almost none of them: five separate mutations of the
production code -- reintroducing the B37 v1 rolling anchor with a one-token
change, deleting the post-evaluation restore, deleting both aliasing copies,
swapping which telemetry field is published, and removing ``commit_anchor``
from the restore -- each left the whole suite green. Two properties caused
that, and both are fixed here:

* the fake inner solve ignored its warm start (its iterate was a pure
  function of the trial coils), so an anchor that advanced per evaluation
  changed nothing observable. :func:`fake_solved_surface` now depends on the
  warm start, which is what a real warm-started Newton does and what makes
  the rolling anchor detectable.
* the fake published one constant for every telemetry field, so a lane that
  published ``G`` where ``iota`` belongs was indistinguishable -- and any
  test written against it would have asserted the constant the fixture
  itself injected. :func:`fake_trial_telemetry` returns mutually distinct,
  coil-derived values instead.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from simsopt_jax_adapters.geo.flat675.policy import FLAT675_OBJECTIVE_TERM_KEYS
from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
    NestedLsOuterAnchor,
    NestedLsOuterTrialReadout,
)

#: The fake inner solve's branch, shared by every module driving a child.
SEED_IOTA = 0.14
SEED_G = 2.0


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


#: Telemetry no real solve produces, mirroring production's own
#: ``NESTED_LS_OUTER_NO_TRIAL_SENTINEL``. A lane that publishes a field
#: nobody set publishes this rather than a plausible number.
FAKE_NO_TRIAL_SENTINEL = -1.0

#: How strongly the fake inner solve's iterate depends on its warm start.
#: Small enough that it perturbs rather than dominates the trial coils'
#: contribution, large enough to be far above float64 noise on these
#: fixtures -- so a rolling anchor shows up as a real difference and not as
#: a last-bit wobble a test would have to guess a tolerance for.
WARM_START_SENSITIVITY = 1.0e-3


def fake_solved_surface(
    coil_dofs: object, warm_start_surface: object
) -> NDArray[np.float64]:
    """The iterate the fake inner solve lands on, warm start included.

    A warm-started Newton lands where it lands partly because of where it
    started. Ignoring that -- returning a pure function of the trial coils,
    which is what this fake used to do -- makes the B37 v1 rolling anchor
    unobservable, because the whole defect IS which warm start the next
    evaluation inherits. With this dependence, an anchor that advanced on a
    rejected trial changes what a later evaluation at the SAME coils
    produces, and a test can see it bitwise.
    """

    coils = np.asarray(coil_dofs, dtype=np.float64).reshape(-1)
    warm = np.asarray(warm_start_surface, dtype=np.float64).reshape(-1)
    return np.array(
        [float(np.sum(coils)) + WARM_START_SENSITIVITY * float(warm[0])],
        dtype=np.float64,
    )


def fake_trial_telemetry(coil_dofs: object) -> dict[str, float]:
    """Five mutually distinguishable per-evaluation numbers.

    Separated by orders of magnitude and derived from the coils, so that a
    lane publishing ``G`` where ``iota`` belongs, or ``inner_grad_l2`` where
    ``adjoint_live_eta`` belongs, yields a number no assertion can mistake
    for the right one. The predecessor returned one injected constant for
    every field, which made a field swap invisible AND would have made any
    test asserting on it a test of its own fixture.
    """

    total = float(np.sum(np.asarray(coil_dofs, dtype=np.float64).reshape(-1)))
    return {
        "iota": SEED_IOTA + 1.0e-4 * total,
        "G": SEED_G + 1.0e-2 * total,
        "inner_iterations": 3 + int(abs(total) * 10.0) % 5,
        "inner_grad_l2": 1.0e-14 * (1.0 + abs(total)),
        "adjoint_live_eta": 2.0e-12 * (1.0 + abs(total)),
    }


def sentinel_anchor() -> NestedLsOuterAnchor:
    """An anchor no real commit produces.

    The default committed anchor for a state under test. A lane that forgets
    to commit then publishes an obviously impossible point rather than a
    plausible one, which is the difference between a test that fails loudly
    and a receipt that ships a wrong number quietly.
    """

    return NestedLsOuterAnchor.at(
        coil_dofs=np.array([FAKE_NO_TRIAL_SENTINEL], dtype=np.float64),
        surface_dofs=np.array([FAKE_NO_TRIAL_SENTINEL], dtype=np.float64),
        iota=FAKE_NO_TRIAL_SENTINEL,
        G=FAKE_NO_TRIAL_SENTINEL,
        schur_lu=None,
    )


class _FakeJaxOuterState:
    """``NestedLsOuterState``'s committed anchor and its trial readout.

    The two records the JAX child touches, and nothing else. ``commit_anchor``
    is the only way to move the anchor, and ``commit_count`` records how many
    times it did — so a test can assert not just WHERE the anchor ended up but
    that it moved exactly as often as scipy accepted, which is the invariant
    the whole transactional lane exists to hold.

    The trial readout starts at :data:`FAKE_NO_TRIAL_SENTINEL`, matching
    production. Zeros would be worse than useless here: ``0.0`` is a perfect
    inner residual and a perfect adjoint residual, so an unset readout would
    PASS a ``<= tol`` gate. The committed anchor defaults to
    :func:`sentinel_anchor` for the same reason; a caller that needs the
    state to start at a specific real point passes one.
    """

    def __init__(
        self,
        anchor: NestedLsOuterAnchor | None = None,
        *,
        inner_substep_legs: tuple[int, ...] = (1,),
        inner_predictor: bool = False,
    ) -> None:
        anchor = sentinel_anchor() if anchor is None else anchor
        self.anchor = anchor
        # The inner-lane policy the child reads to build its ``inner_policy``
        # block. Settable, so a test can drive a NON-stock lane and check the
        # receipt says so -- a fake pinned to the stock values could only ever
        # confirm that a stock run reports stock, which is the half that
        # cannot be wrong.
        self.inner_substep_legs = tuple(inner_substep_legs)
        self.inner_predictor = bool(inner_predictor)
        self.commit_count = 0
        self.last_trial = NestedLsOuterTrialReadout(
            anchor=anchor,
            inner_iterations=int(FAKE_NO_TRIAL_SENTINEL),
            inner_grad_l2=FAKE_NO_TRIAL_SENTINEL,
            adjoint_live_eta=FAKE_NO_TRIAL_SENTINEL,
        )

    def commit_anchor(self, anchor: NestedLsOuterAnchor) -> None:
        self.anchor = anchor
        self.commit_count += 1


class _FakeObjective:
    """The native twin's eight-term objective, replaced by ``J(c) = c.c``."""

    def __init__(self, boozer: _FakeBoozer) -> None:
        self.boozer = boozer

    def evaluate(self) -> tuple[float, dict[str, float], NDArray[np.float64]]:
        coils = np.asarray(self.boozer.biotsavart.x, dtype=np.float64)
        value = objective_at(coils)
        terms = {key: value for key in FLAT675_OBJECTIVE_TERM_KEYS}
        return value, terms, 2.0 * coils
