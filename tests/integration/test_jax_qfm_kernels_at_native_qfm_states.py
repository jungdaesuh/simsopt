"""The ``1_Simple/qfm.py`` mirror's kernels evaluate upstream's physics at a native run's states.

A live native run of upstream's six-call QFM sequence (``QfmSurface``'s L-BFGS-B
penalty and SLSQP exact solves for the volume, toroidal-flux and area labels) at
the bounded resolution supplies nine states: the start and the end of every one
of the six calls (``<label>:{initial,penalty,exact}``).  At each, the native
library's QFM residual, the stage's label and both of their gradients are the
reference, and the mirror's own kernels -- built exactly as the shipped script
builds them -- are evaluated at the same parameters, so nothing here depends on
the mirror's optimizer path.

No tolerance is hand-entered: every comparison is bounded by the worst-case
float64 summation model ``2 (n - 1) u S`` (Higham, *Accuracy and Stability of
Numerical Algorithms*, 2nd ed., section 4.2), with ``n`` the reduction that
quantity performs -- the surface quadrature times the parameter count for the
pure surface labels, and one term per (surface point, coil, coil quadrature
point) for the field-dependent ones.  ``S`` is not measured here, so the budget
scales by the compared value (and, for gradients, by the ambient gradient of the
run), and the factor ``(n - 1)`` over ``lambda sqrt(n)`` stands in for the
unmeasured cancellation amplification.  This is a GROSS-FAILURE bound: it
resolves that the mirror computes upstream's function and gradient at these
states, not a last-bits difference.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from dataclasses import dataclass

import jax
import numpy as np
import pytest
from simsopt.configs.zoo import get_data
from simsopt.field import BiotSavart
from simsopt.geo import Area, QfmSurface, SurfaceRZFourier, ToroidalFlux, Volume
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples.qfm_host_scipy import (
    QFM_SURFACE_RESOLUTION,
    build_qfm_host_kernels,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

#: The shipped script's bounded budget and upstream's solver settings.
MAX_STEPS = 80
TOLERANCE = 1.0e-12
CONSTRAINT_WEIGHT = 1.0
#: The nine ``(label, phase)`` states of the run, in the order the sequence
#: produces them: ``<label>:initial`` is the penalty call's start,
#: ``<label>:penalty`` its end and the exact call's start, ``<label>:exact`` the
#: exact call's end and the next stage's start.
RECORDS: tuple[tuple[str, str], ...] = tuple(
    (label, phase)
    for label in ("volume", "toroidal_flux", "area")
    for phase in ("initial", "penalty", "exact")
)
#: These two labels are pure surface integrals; ``toroidal_flux`` and the QFM
#: residual also reduce the coil field over every coil quadrature point.
SURFACE_ONLY_LABELS = frozenset({"volume", "area"})
#: float64 unit roundoff: half the machine epsilon.
UNIT_ROUNDOFF = float(np.finfo(np.float64).eps) / 2.0


def _worst_case_summation_budget(term_count: int) -> float:
    """Relative budget ``2 (n - 1) u`` for two independently rounded float64 reductions."""
    assert term_count >= 2
    return 2.0 * float(term_count - 1) * UNIT_ROUNDOFF


@dataclass(frozen=True)
class NativeState:
    """One state of the native run and the native physics evaluated at it."""

    label: str
    phase: str
    parameters: np.ndarray
    qfm_value: float
    qfm_gradient: np.ndarray
    label_value: float
    label_gradient: np.ndarray

    @property
    def name(self) -> str:
        return f"{self.label}:{self.phase}"


@dataclass(frozen=True)
class ReplayBudgets:
    """The two reduction sizes this problem performs at one state."""

    field: int
    surface: int

    def term_count(self, label: str) -> int:
        return self.surface if label in SURFACE_ONLY_LABELS else self.field


@dataclass(frozen=True)
class NativeRun:
    states: dict[str, NativeState]
    budgets: ReplayBudgets
    ambient_qfm_gradient: float
    ambient_label_gradient: dict[str, float]


def _fitted_problem() -> tuple[BiotSavart, SurfaceRZFourier]:
    """NCSX coils and upstream's fitted QFM start surface at the bounded resolution."""
    _, _, magnetic_axis, nfp, biotsavart = get_data("ncsx")
    resolution = QFM_SURFACE_RESOLUTION["bounded"]
    surface = SurfaceRZFourier(
        mpol=resolution.order,
        ntor=resolution.order,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(
            0.0, 1.0 / nfp, resolution.quadrature_size, endpoint=False
        ),
        quadpoints_theta=np.linspace(
            0.0, 1.0, resolution.quadrature_size, endpoint=False
        ),
    )
    surface.fit_to_curve(magnetic_axis, 0.2, flip_theta=True)
    return biotsavart, surface


def _record(qfm_surface, label, name: str, phase: str) -> NativeState:
    parameters = np.array(qfm_surface.surface.x, dtype=np.float64, copy=True)
    qfm_value, qfm_gradient = qfm_surface.qfm_objective(parameters, derivatives=1)
    return NativeState(
        label=name,
        phase=phase,
        parameters=parameters,
        qfm_value=float(qfm_value),
        qfm_gradient=np.array(qfm_gradient, dtype=np.float64, copy=True),
        label_value=float(label.J()),
        label_gradient=np.array(
            label.dJ_by_dsurfacecoefficients(), dtype=np.float64, copy=True
        ),
    )


@pytest.fixture(scope="module")
def native_run() -> NativeRun:
    """Upstream's six calls with the native library, and its nine states."""
    biotsavart, surface = _fitted_problem()
    toroidal_field = BiotSavart(biotsavart.coils)
    states: dict[str, NativeState] = {}
    for name in ("volume", "toroidal_flux", "area"):
        if name == "volume":
            label = Volume(surface)
        elif name == "toroidal_flux":
            label = ToroidalFlux(surface, toroidal_field)
        else:
            label = Area(surface)
        qfm_surface = QfmSurface(biotsavart, surface, label, float(label.J()))
        states[f"{name}:initial"] = _record(qfm_surface, label, name, "initial")
        qfm_surface.minimize_qfm_penalty_constraints_LBFGS(
            tol=TOLERANCE, maxiter=MAX_STEPS, constraint_weight=CONSTRAINT_WEIGHT
        )
        states[f"{name}:penalty"] = _record(qfm_surface, label, name, "penalty")
        qfm_surface.minimize_qfm_exact_constraints_SLSQP(
            tol=TOLERANCE, maxiter=MAX_STEPS
        )
        states[f"{name}:exact"] = _record(qfm_surface, label, name, "exact")

    coils = list(biotsavart.coils)
    quadrature = {int(np.asarray(coil.curve.gamma()).shape[0]) for coil in coils}
    assert len(quadrature) == 1, quadrature
    surface_points = int(
        np.asarray(surface.quadpoints_phi).size
        * np.asarray(surface.quadpoints_theta).size
    )
    budgets = ReplayBudgets(
        # The dominant reduction of a coil-field quantity.
        field=surface_points * len(coils) * quadrature.pop(),
        # A pure surface integral evaluates the Fourier series of every
        # parameter at every quadrature point and then sums over the points.
        surface=surface_points * int(np.asarray(surface.x).size),
    )
    # Ambient gradient scales over the run: a gradient component at a converged
    # state is a near-cancelling sum, so its own reduced magnitude does not
    # bound the sum of absolute contributions behind it; the problem's ambient
    # scale does.
    return NativeRun(
        states=states,
        budgets=budgets,
        ambient_qfm_gradient=max(
            float(np.max(np.abs(state.qfm_gradient))) for state in states.values()
        ),
        ambient_label_gradient={
            label: max(
                float(np.max(np.abs(state.label_gradient)))
                for state in states.values()
                if state.label == label
            )
            for label, _phase in RECORDS
        },
    )


@pytest.fixture(scope="module")
def jax_kernels():
    """The mirror's own kernels, compiled once for every state."""
    assert bool(jax.config.read("jax_enable_x64")) is True
    biotsavart, surface = _fitted_problem()
    field = BiotSavartJAX(biotsavart.coils)
    return build_qfm_host_kernels(
        initial_parameters=np.asarray(surface.x, dtype=np.float64),
        quadpoints_phi=np.asarray(surface.quadpoints_phi, dtype=np.float64),
        quadpoints_theta=np.asarray(surface.quadpoints_theta, dtype=np.float64),
        coil_set_spec=field.coil_set_spec_from_dofs(
            explicit_device_array(
                field.x, dtype=np.float64, device=get_runtime_jax_device()
            )
        ),
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
    )


def _assert_quantity(
    what: str,
    *,
    observed_value: float,
    observed_gradient: np.ndarray,
    reference_value: float,
    reference_gradient: np.ndarray,
    ambient_gradient: float,
    term_count: int,
) -> None:
    """Compare one value/gradient pair with the native reference."""
    budget = _worst_case_summation_budget(term_count)
    value_gap = abs(observed_value - reference_value)
    allowed_value = budget * abs(reference_value)
    assert value_gap <= allowed_value, (
        f"{what}: value differs from the native library by {value_gap:.6e}, "
        f"above the derived budget {allowed_value:.6e}"
    )
    gradient_gap = float(np.max(np.abs(observed_gradient - reference_gradient)))
    allowed_gradient = budget * ambient_gradient
    assert gradient_gap <= allowed_gradient, (
        f"{what}: gradient differs from the native library by "
        f"{gradient_gap:.6e}, above the derived budget {allowed_gradient:.6e}"
    )


@pytest.mark.parametrize(
    "record", RECORDS, ids=[f"{label}:{phase}" for label, phase in RECORDS]
)
def test_jax_kernels_match_the_native_state(
    native_run: NativeRun,
    jax_kernels,
    record: tuple[str, str],
) -> None:
    """The mirror's own kernels at one state of the native run."""
    label, phase = record
    state = native_run.states[f"{label}:{phase}"]
    qfm_value, qfm_gradient = jax_kernels.host_qfm()(state.parameters)
    _assert_quantity(
        f"jax {state.name} qfm",
        observed_value=float(qfm_value),
        observed_gradient=np.asarray(qfm_gradient, dtype=np.float64),
        reference_value=state.qfm_value,
        reference_gradient=state.qfm_gradient,
        ambient_gradient=native_run.ambient_qfm_gradient,
        term_count=native_run.budgets.field,
    )
    label_value, label_gradient = jax_kernels.host_label(state.label)(state.parameters)
    _assert_quantity(
        f"jax {state.name} label",
        observed_value=float(label_value),
        observed_gradient=np.asarray(label_gradient, dtype=np.float64),
        reference_value=state.label_value,
        reference_gradient=state.label_gradient,
        ambient_gradient=native_run.ambient_label_gradient[state.label],
        term_count=native_run.budgets.term_count(state.label),
    )
