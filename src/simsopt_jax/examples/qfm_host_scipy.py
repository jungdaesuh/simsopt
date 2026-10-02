"""Faithful host-SciPy QFM control over compiled JAX QFM physics.

The official script runs six SciPy calls -- one ``L-BFGS-B`` penalty call and
one ``SLSQP`` equality-constraint call for each of volume, toroidal flux and
area (``simsopt/geo/qfmsurface.py``).  This module reproduces those six calls
over JAX physics and crosses the host/device boundary only through the
repository's owners:

* the penalty call is handed to ``solve.dispatch.minimize``, the owner of the
  host-SciPy driver boundary, over a device objective;
* the exact-constraint call cannot be: ``dispatch`` has no SLSQP driver and no
  way to carry an equality constraint, so that call stays here and materializes
  its packets through ``runtime.host_boundary`` / ``backend.dtypes`` instead of
  calling ``jax.device_put`` / ``jax.device_get`` itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Final, Literal, Mapping

import jax
import numpy as np
from scipy.optimize import OptimizeResult, minimize

from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.core.qfm_solver import (
    qfm_label_jax_from_dofs,
    qfm_residual_jax_from_dofs,
)
from simsopt_jax.core.specs import (
    GroupedCoilSetSpec,
    SurfaceRZFourierSpec,
    host_resident_spec,
)
from simsopt_jax.core.surface_rzfourier import surface_rz_fourier_spec_from_dofs
from simsopt_jax.examples.execution import ExecutionScale
from simsopt_jax.runtime.host_boundary import host_tree_after_ready
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.dispatch import minimize as dispatch_minimize
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.scipy.contracts import ScipyLBFGSBOptions

QfmLabel = Literal["volume", "toroidal_flux", "area"]
QFM_HOST_SCIPY_DRIVER = "scipy_lbfgsb_slsqp_qfm_sequence"
QFM_PENALTY_DRIVER: Final[Driver] = Driver.SCIPY_LBFGSB
QFM_EXACT_METHOD = "SLSQP"
QFM_LBFGSB_HISTORY = 200
QFM_TOROIDAL_FLUX_IDX = 0
#: The three reference labels in the order ``1_Simple/qfm.py`` solves them.
QFM_LABELS: Final[tuple[QfmLabel, ...]] = ("volume", "toroidal_flux", "area")


@dataclass(frozen=True)
class QfmSurfaceResolution:
    """Fourier order and quadrature size of one QFM surface resolution."""

    order: int
    quadrature_size: int


#: Surface resolution per execution scale, the single source of this contract.
#: ``native_default`` is upstream's own: ``1_Simple/qfm.py`` sets
#: ``mpol = ntor = 5`` and ``np.linspace(..., 25, endpoint=False)`` for both
#: angles. ``bounded`` is the branch's smoke resolution. The shipped mirror
#: and the strict-transfer test both read this table.
QFM_SURFACE_RESOLUTION: Final[Mapping[ExecutionScale, QfmSurfaceResolution]] = (
    MappingProxyType(
        {
            "native_default": QfmSurfaceResolution(order=5, quadrature_size=25),
            "bounded": QfmSurfaceResolution(order=1, quadrature_size=6),
        }
    )
)

DeviceValueAndGradient = Callable[[jax.Array], tuple[jax.Array, jax.Array]]
HostValueAndGradient = Callable[[np.ndarray], tuple[float, np.ndarray]]


@dataclass(frozen=True)
class QfmHostState:
    """JAX QFM metrics materialized for one accepted host parameter vector."""

    parameters: np.ndarray
    qfm_value: float
    qfm_gradient: np.ndarray
    label_value: float
    label_gradient: np.ndarray
    label_residual_abs: float


@dataclass(frozen=True)
class QfmHostStageResult:
    """One reference penalty/SLSQP pair and its actual SciPy results.

    ``penalty_optimizer`` comes back from the owned driver boundary and
    ``exact_optimizer`` straight from SciPy; both publish ``x``, ``fun``,
    ``status``, ``message``, ``nit``, ``nfev``, ``njev`` and ``success``, and
    the two annotations collapse into one once ``dispatch`` grows an SLSQP
    driver.
    """

    label: QfmLabel
    target: float
    initial: QfmHostState
    penalty: QfmHostState
    exact: QfmHostState
    penalty_optimizer: OptimizerResult
    exact_optimizer: OptimizeResult
    volume_persistence_objective: float


@dataclass(frozen=True)
class QfmHostSequenceResult:
    """The three consecutive reference-label pairs from ``1_Simple/qfm.py``."""

    initial_volume: float
    volume: QfmHostStageResult
    toroidal_flux: QfmHostStageResult
    area: QfmHostStageResult


@dataclass(frozen=True)
class QfmHostKernels:
    """Compiled QFM physics for one surface and coil configuration.

    ``qfm`` and every entry of ``labels`` take device surface DOFs and return
    the device ``(value, gradient)`` packet of one physics quantity; the host
    wrappers built from them are what SciPy sees.
    """

    qfm: DeviceValueAndGradient
    labels: Mapping[QfmLabel, DeviceValueAndGradient]

    def host_qfm(self) -> HostValueAndGradient:
        """The QFM residual and its gradient, as host values."""
        return _host_value_and_gradient(self.qfm)

    def host_label(self, label: QfmLabel) -> HostValueAndGradient:
        """One reference label and its gradient, as host values."""
        return _host_value_and_gradient(self.labels[label])

    def penalty(
        self,
        *,
        label: QfmLabel,
        constraint_weight: float,
    ) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        """Upstream's penalty packet, in upstream's operand order, on device.

        ``qfmsurface.qfm_penalty_constraints`` forms ``r + 0.5*w*rl**2`` and
        ``dr + w*rl*dl`` from two independently evaluated physics quantities;
        this is the same expression over the same two kernels, evaluated where
        the physics already is so the owned driver boundary sees one device
        objective. The stage target is an OPERAND, never a capture.
        """
        qfm = self.qfm
        label_kernel = self.labels[label]

        def penalty_packet(
            parameters: jax.Array,
            target: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            qfm_value, qfm_gradient = qfm(parameters)
            label_value, label_gradient = label_kernel(parameters)
            residual = label_value - target
            return (
                qfm_value + 0.5 * constraint_weight * residual**2,
                qfm_gradient + constraint_weight * residual * label_gradient,
            )

        return jax.jit(penalty_packet)


def _host_value_and_gradient(
    device_value_and_gradient: DeviceValueAndGradient,
) -> HostValueAndGradient:
    """Materialize one SciPy callback packet through the boundary owners.

    ``explicit_device_array`` (H2D owner) places the host parameters on the
    runtime device and ``host_tree_after_ready`` (D2H owner) brings the value
    and the gradient back in a single crossing.
    """

    def evaluate(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        placed = explicit_device_array(parameters, dtype=np.float64)
        value, gradient = host_tree_after_ready(
            device_value_and_gradient(placed),
            dtype=np.float64,
        )
        return float(value), gradient

    return evaluate


def _device_value_and_gradient(
    value: Callable[[jax.Array], jax.Array],
) -> DeviceValueAndGradient:
    return jax.jit(jax.value_and_grad(value))


def build_qfm_host_kernels(
    *,
    initial_parameters: np.ndarray,
    quadpoints_phi: np.ndarray,
    quadpoints_theta: np.ndarray,
    coil_set_spec: GroupedCoilSetSpec,
    mpol: int,
    ntor: int,
    nfp: int,
    stellsym: bool,
) -> QfmHostKernels:
    """Compile the QFM residual and the three reference labels for one surface.

    Every kernel built here -- and every penalty kernel
    :meth:`QfmHostKernels.penalty` later builds on top of them -- CAPTURES the
    surface and coil specs in its closure rather than receiving them as
    operands, so both must be host-resident. XLA turns a captured concrete
    array into an MLIR literal by copying it back to the host, once per
    lowering, and the ``jax-gpu`` lane refuses that copy: the observed failure
    was ``Disallowed device-to-host transfer: shape=(25), dtype=F64,
    device=cuda:0`` -- the official surface's ``quadpoints_phi`` -- raised from
    ``_array_mlir_constant_handler``. ``host_resident_spec`` is the
    repository's single owner of that rule and routes the one build-time
    read-back through the audited host boundary.
    """
    surface_spec = host_resident_spec(
        _surface_spec(
            initial_parameters=initial_parameters,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=stellsym,
        )
    )
    captured_coil_set_spec = host_resident_spec(coil_set_spec)

    def qfm(parameters: jax.Array) -> jax.Array:
        return qfm_residual_jax_from_dofs(
            surface_spec, parameters, captured_coil_set_spec
        )

    return QfmHostKernels(
        qfm=_device_value_and_gradient(qfm),
        labels={
            label: _device_value_and_gradient(
                _label_value(surface_spec, captured_coil_set_spec, label=label)
            )
            for label in QFM_LABELS
        },
    )


def _label_value(
    surface_spec: SurfaceRZFourierSpec,
    coil_set_spec: GroupedCoilSetSpec,
    *,
    label: QfmLabel,
) -> Callable[[jax.Array], jax.Array]:
    def value(parameters: jax.Array) -> jax.Array:
        return qfm_label_jax_from_dofs(
            surface_spec,
            parameters,
            coil_set_spec,
            label=label,
            toroidal_flux_idx=QFM_TOROIDAL_FLUX_IDX,
        )

    return value


def _surface_spec(
    *,
    initial_parameters: np.ndarray,
    quadpoints_phi: np.ndarray,
    quadpoints_theta: np.ndarray,
    mpol: int,
    ntor: int,
    nfp: int,
    stellsym: bool,
) -> SurfaceRZFourierSpec:
    return surface_rz_fourier_spec_from_dofs(
        explicit_device_array(initial_parameters, dtype=np.float64),
        quadpoints_phi=explicit_device_array(quadpoints_phi, dtype=np.float64),
        quadpoints_theta=explicit_device_array(quadpoints_theta, dtype=np.float64),
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
    )


def _state(
    parameters: np.ndarray,
    *,
    target: float,
    qfm_value_and_gradient: HostValueAndGradient,
    label_value_and_gradient: HostValueAndGradient,
) -> QfmHostState:
    qfm_value, qfm_gradient = qfm_value_and_gradient(parameters)
    label_value, label_gradient = label_value_and_gradient(parameters)
    return QfmHostState(
        parameters=np.asarray(parameters, dtype=np.float64),
        qfm_value=qfm_value,
        qfm_gradient=qfm_gradient,
        label_value=label_value,
        label_gradient=label_gradient,
        label_residual_abs=abs(label_value - target),
    )


def solve_qfm_host_scipy_stage(
    initial_parameters: np.ndarray,
    *,
    kernels: QfmHostKernels,
    label: QfmLabel,
    initial_volume: float,
    tolerance: float,
    max_steps: int,
    target: float | None = None,
    constraint_weight: float = 1.0,
) -> QfmHostStageResult:
    """One official QFM stage from ``initial_parameters``: penalty, then exact.

    ``target`` defaults to the label at that start, which is the official rule,
    and is given explicitly to replay one stage from a captured state and a
    captured target; SLSQP starts at the L-BFGS-B endpoint. ``initial_volume``
    is the sequence's own start volume and only feeds volume persistence.
    """
    qfm_value_and_gradient = kernels.host_qfm()
    label_value_and_gradient = kernels.host_label(label)
    volume_value_and_gradient = kernels.host_label("volume")
    if target is None:
        stage_target, _ = label_value_and_gradient(initial_parameters)
    else:
        stage_target = float(target)
    initial = _state(
        initial_parameters,
        target=stage_target,
        qfm_value_and_gradient=qfm_value_and_gradient,
        label_value_and_gradient=label_value_and_gradient,
    )

    penalty_packet = kernels.penalty(
        label=label,
        constraint_weight=constraint_weight,
    )
    target_device = explicit_device_array(stage_target, dtype=np.float64)

    def penalty_value_and_gradient(
        parameters: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return penalty_packet(parameters, target_device)

    # The owner of this boundary. ``native_matched`` expands one ``tol`` into
    # ``ftol`` and ``gtol`` and leaves ``maxfun``/``maxls`` at SciPy's own
    # defaults, which is exactly the option set ``qfmsurface.py`` names.
    penalty_optimizer = dispatch_minimize(
        penalty_value_and_gradient,
        explicit_device_array(initial.parameters, dtype=np.float64),
        driver=QFM_PENALTY_DRIVER,
        options=ScipyLBFGSBOptions.native_matched(
            maxiter=max_steps,
            maxcor=QFM_LBFGSB_HISTORY,
            tol=tolerance,
        ),
    )
    penalty = _state(
        penalty_optimizer.x,
        target=stage_target,
        qfm_value_and_gradient=qfm_value_and_gradient,
        label_value_and_gradient=label_value_and_gradient,
    )

    def constraint_value(parameters: np.ndarray) -> float:
        label_value, _ = label_value_and_gradient(parameters)
        residual = label_value - stage_target
        return 0.5 * residual * residual

    def constraint_gradient(parameters: np.ndarray) -> np.ndarray:
        label_value, label_gradient = label_value_and_gradient(parameters)
        return (label_value - stage_target) * label_gradient

    # Not routed through ``dispatch``: it has no SLSQP driver and no way to
    # carry ``constraints``. See ``SHARED-FILE-REQUESTS.md`` in this wave.
    exact_optimizer = minimize(
        qfm_value_and_gradient,
        penalty.parameters,
        jac=True,
        method=QFM_EXACT_METHOD,
        constraints=[
            {
                "type": "eq",
                "fun": constraint_value,
                "jac": constraint_gradient,
            }
        ],
        options={"maxiter": max_steps, "ftol": tolerance},
    )
    exact = _state(
        exact_optimizer.x,
        target=stage_target,
        qfm_value_and_gradient=qfm_value_and_gradient,
        label_value_and_gradient=label_value_and_gradient,
    )
    volume_value, _ = volume_value_and_gradient(exact.parameters)
    volume_residual = volume_value - initial_volume
    return QfmHostStageResult(
        label=label,
        target=stage_target,
        initial=initial,
        penalty=penalty,
        exact=exact,
        penalty_optimizer=penalty_optimizer,
        exact_optimizer=exact_optimizer,
        volume_persistence_objective=0.5 * volume_residual * volume_residual,
    )


def solve_qfm_host_scipy_sequence(
    initial_parameters: np.ndarray,
    *,
    kernels: QfmHostKernels,
    max_steps: int,
    tolerance: float,
    constraint_weight: float = 1.0,
) -> QfmHostSequenceResult:
    """Run the official QFM L-BFGS-B/SLSQP stage sequence over JAX physics.

    Volume, then toroidal flux, then area; each stage starts where the previous
    one ended and takes its own label target there. ``kernels`` comes from
    :func:`build_qfm_host_kernels`, which owns the surface and coil geometry.
    """
    host_parameters = np.asarray(initial_parameters, dtype=np.float64)
    initial_volume, _ = kernels.host_label("volume")(host_parameters)
    stages: list[QfmHostStageResult] = []
    parameters = host_parameters
    for label in QFM_LABELS:
        stage = solve_qfm_host_scipy_stage(
            parameters,
            kernels=kernels,
            label=label,
            initial_volume=initial_volume,
            tolerance=tolerance,
            max_steps=max_steps,
            constraint_weight=constraint_weight,
        )
        stages.append(stage)
        parameters = stage.exact.parameters
    volume, toroidal_flux, area = stages
    return QfmHostSequenceResult(
        initial_volume=initial_volume,
        volume=volume,
        toroidal_flux=toroidal_flux,
        area=area,
    )


__all__ = [
    "QFM_EXACT_METHOD",
    "QFM_HOST_SCIPY_DRIVER",
    "QFM_LABELS",
    "QFM_LBFGSB_HISTORY",
    "QFM_PENALTY_DRIVER",
    "QFM_SURFACE_RESOLUTION",
    "QfmHostKernels",
    "QfmHostSequenceResult",
    "QfmHostStageResult",
    "QfmHostState",
    "QfmSurfaceResolution",
    "build_qfm_host_kernels",
    "solve_qfm_host_scipy_sequence",
    "solve_qfm_host_scipy_stage",
]
