"""Both qfm lanes evaluate upstream's physics at upstream's own states.

``native-qfm`` is judged at ``native_default`` by an endpoint quality band
(``examples/jax/parity/official_quality_bands.py``), and under a band the
arbiter makes every lane-versus-lane comparison informational.  The same-state
evidence therefore has to be carried by a tracked test, and the one this case
had compared the JAX lane at the START state only.

This file replays the official run's OWN states.  The tracked fixture
``examples/jax/parity/official_reference`` records the surface parameters at the
start and at the end of every one of the six SciPy calls
(``<label>:{initial,penalty,exact}:parameters``), and at each of them the QFM
residual, that stage's reference label, and BOTH of their gradients.  The native
lane and the JAX lane are evaluated at those parameters and compared with
upstream, so nothing here depends on either lane's optimizer path.

Only gradients the fixture actually holds are compared.  SciPy's SLSQP
``result.jac`` is NOT the gradient at ``result.x`` -- it differs from it by 57 %,
149 % and 149 % on BOTH official builds
(``A/investigations/end-state-replay/REPORT.md``) -- so the recorded provider
Jacobians are evidence about nobody; ``*:qfm_gradient`` and ``*:label_gradient``
are, because the capture evaluated them itself at the recorded state.

What the bounds here do and do not resolve
------------------------------------------
No tolerance is hand-entered: every comparison is bounded by the WORST-CASE
float64 summation model of ``tests/official_state_budget.py``,
``2 (n - 1) u S``, with ``n`` the reduction that quantity performs -- the
surface quadrature times the parameter count for the pure surface labels, and
``biot_savart_term_count`` for the field-dependent ones.

That model, not the probabilistic one its sibling replay uses for the flux, is
the right one HERE because this file never measures the ``S`` of the sums it
bounds.  It scales the budget by the compared value (and, for gradients, by the
ambient gradient of the run), while the contributions behind both are signed --
a label gradient sums Fourier terms of either sign, and the QFM residual's field
reduction is the same one the stage-two replay measures a cancellation
amplification of 11 to 2600 for.  The factor ``(n - 1)`` over
``lambda sqrt(n)``, 256x at this ``n``, is what stands in for that unmeasured
amplification.

So this is a GROSS-FAILURE bound, and the numbers say by how much.  Measured
over all 36 comparisons of the nine states, the worst usage of the asserted
budget is 5.03e-4 of it -- about 2000x slack (worst comparison: the JAX lane's
QFM value at ``toroidal_flux:penalty``, gap 6.47e-20 against a budget of
1.29e-16).  The same
measurements use 0.129 of the probabilistic budget at the same ``n``, i.e. the
tighter model would be met, but with 7.8x headroom against an amplification
measured at up to 2600x for this reduction elsewhere, so it is not asserted.
What this file resolves is that both lanes compute upstream's function and
upstream's gradient at upstream's own states; it does not resolve a last-bits
difference between them.  Measured at ``volume:initial``, scaling the state's
own parameters by ``1 + 2^-32`` (2.3e-10 relative) is rejected by the QFM value
and ``1 + 2^-36`` (1.5e-11) by the label value, while ``1 + 2^-34`` and
``1 + 2^-40`` respectively are accepted.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import native_qfm as case
from examples.jax.parity.input_bundle import InputBundle, load_input_bundle
from examples.jax.parity.official_reference import (
    OfficialReference,
    array_sha256,
    load_official_reference,
)
from simsopt.field import BiotSavart
from simsopt.geo import Area, QfmSurface, ToroidalFlux, Volume
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples.qfm_host_scipy import build_qfm_host_kernels
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

import jax

# venv site-packages/tests shadows the repo tests package, so the helper is
# imported as a top-level module from the tests/ directory.
_TESTS_ROOT = str(Path(__file__).resolve().parents[1])
if _TESTS_ROOT not in sys.path:
    sys.path.append(_TESTS_ROOT)
from official_state_budget import (  # noqa: E402
    biot_savart_term_count,
    worst_case_summation_budget,
)

CASE_ID = "native-qfm"
OFFICIAL: OfficialReference = load_official_reference(CASE_ID)

#: The nine ``(label, phase)`` records of the official run, in the order the
#: script produces them.  They are the start and the end state of all six
#: provider calls: ``<label>:initial`` is the penalty call's start,
#: ``<label>:penalty`` its end and the exact call's start, ``<label>:exact`` the
#: exact call's end and the next stage's start.
OFFICIAL_RECORDS: tuple[tuple[str, str], ...] = tuple(
    (label, phase)
    for label in ("volume", "toroidal_flux", "area")
    for phase in ("initial", "penalty", "exact")
)
#: These two labels are pure surface integrals; ``toroidal_flux`` and the QFM
#: residual also reduce the coil field over every coil quadrature point.
SURFACE_ONLY_LABELS = frozenset({"volume", "area"})


@dataclass(frozen=True)
class OfficialState:
    """One official state and the official physics recorded at it."""

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


def _official_scalar(key: str) -> float:
    value = OFFICIAL.scalar(key)
    assert isinstance(value, float), key
    return value


def official_states() -> tuple[OfficialState, ...]:
    """Every recorded official state, read from the tracked fixture."""
    return tuple(
        OfficialState(
            label=label,
            phase=phase,
            parameters=OFFICIAL.array(f"{label}:{phase}:parameters"),
            qfm_value=_official_scalar(f"{label}:{phase}:qfm_value"),
            qfm_gradient=OFFICIAL.array(f"{label}:{phase}:qfm_gradient"),
            label_value=_official_scalar(f"{label}:{phase}:label_value"),
            label_gradient=OFFICIAL.array(f"{label}:{phase}:label_gradient"),
        )
        for label, phase in OFFICIAL_RECORDS
    )


STATES: tuple[OfficialState, ...] = official_states()
#: Ambient gradient scales, from the official record itself.  A gradient
#: component at a converged state is a near-cancelling sum -- the QFM gradient
#: falls by five orders along the run -- so its own reduced magnitude does not
#: bound the sum of absolute contributions behind it; the problem's ambient
#: scale does.
AMBIENT_QFM_GRADIENT: float = max(
    float(np.max(np.abs(state.qfm_gradient))) for state in STATES
)
AMBIENT_LABEL_GRADIENT: dict[str, float] = {
    label: max(
        float(np.max(np.abs(state.label_gradient)))
        for state in STATES
        if state.label == label
    )
    for label, _phase in OFFICIAL_RECORDS
}


@dataclass(frozen=True)
class ReplayBudgets:
    """The two reduction sizes this problem performs at one state."""

    field: int
    surface: int

    def term_count(self, label: str) -> int:
        return self.surface if label in SURFACE_ONLY_LABELS else self.field


@pytest.fixture(scope="module")
def native_default_input(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[InputBundle, dict[str, np.ndarray]]:
    """The case's own shipped-scale input bundle, built once."""
    root = Path(tmp_path_factory.mktemp("qfm-official-states")) / "inputs"
    bundle = case.create_input(root, "native_default")
    _, arrays = load_input_bundle(root, bundle)
    return bundle, arrays


@pytest.fixture(scope="module")
def budgets(
    native_default_input: tuple[InputBundle, dict[str, np.ndarray]],
) -> ReplayBudgets:
    bundle, arrays = native_default_input
    _, _, coils = _components(bundle, arrays)
    quadrature = {int(np.asarray(coil.curve.gamma()).shape[0]) for coil in coils}
    assert len(quadrature) == 1, quadrature
    surface_points = int(
        arrays["quadrature_phi"].size * arrays["quadrature_theta"].size
    )
    return ReplayBudgets(
        # The dominant reduction of a coil-field quantity.
        field=biot_savart_term_count(
            surface_points=surface_points,
            coils=len(coils),
            curve_quadrature=quadrature.pop(),
        ),
        # A pure surface integral evaluates the Fourier series of every
        # parameter at every quadrature point and then sums over the points.
        surface=surface_points * int(arrays["initial_parameters"].size),
    )


@pytest.fixture(scope="module")
def native_problem(
    native_default_input: tuple[InputBundle, dict[str, np.ndarray]],
) -> tuple[object, object, BiotSavart]:
    """The case's own native construction, built once for every state."""
    bundle, arrays = native_default_input
    biotsavart, surface, _coils = _components(bundle, arrays)
    return biotsavart, surface, BiotSavart(biotsavart.coils)


@pytest.fixture(scope="module")
def jax_kernels(native_default_input: tuple[InputBundle, dict[str, np.ndarray]]):
    """The JAX lane's own kernels, compiled once for every state."""
    assert bool(jax.config.read("jax_enable_x64")) is True
    bundle, arrays = native_default_input
    biotsavart, surface, _coils = _components(bundle, arrays)
    device = get_runtime_jax_device()
    field = BiotSavartJAX(biotsavart.coils)
    return build_qfm_host_kernels(
        initial_parameters=arrays["initial_parameters"],
        quadpoints_phi=arrays["quadrature_phi"],
        quadpoints_theta=arrays["quadrature_theta"],
        coil_set_spec=field.coil_set_spec_from_dofs(
            explicit_device_array(field.x, dtype=np.float64, device=device)
        ),
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
    )


def _components(bundle: InputBundle, arrays: dict[str, np.ndarray]):
    """The case's own construction: its BiotSavart, its surface, its coils."""
    biotsavart, surface, _fingerprint = case._problem_components(bundle, arrays)
    return biotsavart, surface, list(biotsavart.coils)


def _native_label(state_label: str, surface, toroidal_field: BiotSavart):
    if state_label == "volume":
        return Volume(surface)
    if state_label == "toroidal_flux":
        return ToroidalFlux(surface, toroidal_field)
    return Area(surface)


def _assert_quantity(
    what: str,
    *,
    observed_value: float,
    observed_gradient: np.ndarray,
    official_value: float,
    official_gradient: np.ndarray,
    ambient_gradient: float,
    term_count: int,
) -> None:
    """Compare one value/gradient pair with the official record."""
    budget = worst_case_summation_budget(term_count)
    value_gap = abs(observed_value - official_value)
    allowed_value = budget * abs(official_value)
    assert value_gap <= allowed_value, (
        f"{what}: value differs from the official record by {value_gap:.6e}, "
        f"above the derived budget {allowed_value:.6e}"
    )
    gradient_gap = float(np.max(np.abs(observed_gradient - official_gradient)))
    allowed_gradient = budget * ambient_gradient
    assert gradient_gap <= allowed_gradient, (
        f"{what}: gradient differs from the official record by "
        f"{gradient_gap:.6e}, above the derived budget {allowed_gradient:.6e}"
    )


def test_the_recorded_states_are_the_endpoints_of_the_six_provider_calls() -> None:
    """Nine records, seven distinct states, one per provider-call endpoint.

    Each stage starts where the previous one ended, so three of the nine
    recorded parameter vectors are repeats -- asserted bitwise here, because
    the mapping from records to provider calls depends on it.
    """
    assert len(OFFICIAL.provider_calls) == 6
    by_name = {state.name: state for state in STATES}
    for earlier, later in (
        ("volume:exact", "toroidal_flux:initial"),
        ("toroidal_flux:exact", "area:initial"),
    ):
        assert array_sha256(by_name[earlier].parameters) == array_sha256(
            by_name[later].parameters
        )
    distinct = {array_sha256(state.parameters) for state in STATES}
    assert len(distinct) == 7
    # The run's published endpoints are the first and the last of the chain.
    assert array_sha256(by_name["volume:initial"].parameters) == array_sha256(
        OFFICIAL.array("initial:parameters")
    )
    assert array_sha256(by_name["area:exact"].parameters) == array_sha256(
        OFFICIAL.array("final:parameters")
    )


def test_the_case_reproduces_upstreams_own_start_vector(
    native_default_input: tuple[InputBundle, dict[str, np.ndarray]],
    budgets: ReplayBudgets,
) -> None:
    """The case builds upstream's start surface, to the construction's roundoff.

    Not bitwise: ``SurfaceRZFourier.fit_to_curve`` runs a least-squares solve,
    and two builds of it differ in the last bits (measured: 4.44e-16 absolute
    over the 121 parameters, on components whose own magnitude is 1e-17).  The
    bound is this file's own rule -- the float64 summation budget of a surface
    reduction -- against the parameter scale, and the states replayed below are
    upstream's regardless, because they are read from the fixture.
    """
    _bundle, arrays = native_default_input
    official = OFFICIAL.array("initial:parameters")

    gap = float(np.max(np.abs(arrays["initial_parameters"] - official)))
    allowed = worst_case_summation_budget(budgets.surface) * float(
        np.max(np.abs(official))
    )
    assert gap <= allowed, (
        f"the case's start vector differs from upstream's by {gap:.6e}, above "
        f"the derived budget {allowed:.6e}"
    )


@pytest.mark.parametrize("state", STATES, ids=[state.name for state in STATES])
def test_native_lane_matches_the_official_state(
    native_problem: tuple[object, object, BiotSavart],
    budgets: ReplayBudgets,
    state: OfficialState,
) -> None:
    """The branch's native physics at one official record."""
    biotsavart, surface, toroidal_field = native_problem
    label = _native_label(state.label, surface, toroidal_field)
    qfm_surface = QfmSurface(
        biotsavart, surface, label, _official_scalar(f"{state.label}:target")
    )
    # ``qfm_objective`` sets ``surface.x`` (``qfmsurface.py:73``), so the label
    # below is read at the official state too.
    qfm_value, qfm_gradient = qfm_surface.qfm_objective(state.parameters, derivatives=1)

    _assert_quantity(
        f"native {state.name} qfm",
        observed_value=float(qfm_value),
        observed_gradient=np.asarray(qfm_gradient, dtype=np.float64),
        official_value=state.qfm_value,
        official_gradient=state.qfm_gradient,
        ambient_gradient=AMBIENT_QFM_GRADIENT,
        term_count=budgets.field,
    )
    _assert_quantity(
        f"native {state.name} label",
        observed_value=float(label.J()),
        observed_gradient=np.asarray(
            label.dJ_by_dsurfacecoefficients(), dtype=np.float64
        ),
        official_value=state.label_value,
        official_gradient=state.label_gradient,
        ambient_gradient=AMBIENT_LABEL_GRADIENT[state.label],
        term_count=budgets.term_count(state.label),
    )


@pytest.mark.parametrize("state", STATES, ids=[state.name for state in STATES])
def test_jax_lane_matches_the_official_state(
    jax_kernels,
    budgets: ReplayBudgets,
    state: OfficialState,
) -> None:
    """The JAX lane's own kernels at one official record.

    The kernels are built exactly as ``native_qfm._jax`` builds them, so this is
    the program the JAX lane hands SciPy, not a re-derivation of the physics.
    """
    qfm_value, qfm_gradient = jax_kernels.host_qfm()(state.parameters)
    _assert_quantity(
        f"jax {state.name} qfm",
        observed_value=float(qfm_value),
        observed_gradient=np.asarray(qfm_gradient, dtype=np.float64),
        official_value=state.qfm_value,
        official_gradient=state.qfm_gradient,
        ambient_gradient=AMBIENT_QFM_GRADIENT,
        term_count=budgets.field,
    )
    label_value, label_gradient = jax_kernels.host_label(state.label)(state.parameters)
    _assert_quantity(
        f"jax {state.name} label",
        observed_value=float(label_value),
        observed_gradient=np.asarray(label_gradient, dtype=np.float64),
        official_value=state.label_value,
        official_gradient=state.label_gradient,
        ambient_gradient=AMBIENT_LABEL_GRADIENT[state.label],
        term_count=budgets.term_count(state.label),
    )
