"""No QFM kernel may capture a device array in its closure.

``jax.jit`` lowers a concrete ``jax.Array`` that a traced function *captures*
into an MLIR literal, which XLA produces by copying the array back to the host
once per lowering. On a real device that copy is a device-to-host transfer, and
the ``jax-gpu`` lane runs under ``jax.transfer_guard("disallow")``, which
refuses it. The observed failure was ``jax.errors.JaxRuntimeError:
INVALID_ARGUMENT: Disallowed device-to-host transfer: shape=(25), dtype=F64,
device=cuda:0`` raised from ``_array_mlir_constant_handler`` while lowering the
first QFM label kernel -- shape ``(25,)`` is the official surface's
``quadpoints_phi``.

A single-memory CPU backend cannot observe that copy, so the guard alone is not
a CPU-effective pin. The placement of the captured constants is: this file
lowers every kernel the module builds and reads the closed-over constants out
of the jaxpr, which fails on CPU exactly when a device array is captured.

``simsopt_jax.core.specs.host_resident_spec`` is the repository's single owner
of this rule and states it in its own docstring; the sibling pins for the other
two capture sites are ``tests/jax/core/test_coil_dof_extraction_spec_placement.py``
and ``tests/geo/test_captured_coil_spec_placement.py``.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from typing import Callable

import jax
import jax.extend.core as extend_core
import numpy as np
from simsopt.configs.zoo import get_data
from simsopt.geo import SurfaceRZFourier
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples.qfm_host_scipy import (
    QFM_LABELS,
    QFM_SURFACE_RESOLUTION,
    QfmHostKernels,
    build_qfm_host_kernels,
)
from simsopt_jax.runtime.host_boundary import disallow_host_transfers
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

#: The constraint weight ``1_Simple/qfm.py`` uses for every penalty call.
CONSTRAINT_WEIGHT = 1.0
#: The branch's smoke resolution: the placement property is a property of how
#: the kernels are built, not of the problem size, and the bounded surface
#: compiles in seconds.
BOUNDED_RESOLUTION = QFM_SURFACE_RESOLUTION["bounded"]


def _bounded_kernels() -> tuple[np.ndarray, QfmHostKernels]:
    """The bounded NCSX QFM kernels, built exactly as the lanes build them."""
    _curves, _currents, magnetic_axis, nfp, native_field = get_data("ncsx")
    quadrature_size = BOUNDED_RESOLUTION.quadrature_size
    quadrature_phi = np.linspace(0.0, 1.0 / nfp, quadrature_size, endpoint=False)
    quadrature_theta = np.linspace(0.0, 1.0, quadrature_size, endpoint=False)
    surface = SurfaceRZFourier(
        mpol=BOUNDED_RESOLUTION.order,
        ntor=BOUNDED_RESOLUTION.order,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=quadrature_phi,
        quadpoints_theta=quadrature_theta,
    )
    surface.fit_to_curve(magnetic_axis, 0.2, flip_theta=True)
    field = BiotSavartJAX(native_field.coils)
    coil_set_spec = field.coil_set_spec_from_dofs(
        explicit_device_array(
            field.x,
            dtype=np.float64,
            device=get_runtime_jax_device(),
        )
    )
    initial_parameters = np.asarray(surface.get_dofs(), dtype=np.float64)
    kernels = build_qfm_host_kernels(
        initial_parameters=initial_parameters,
        quadpoints_phi=quadrature_phi,
        quadpoints_theta=quadrature_theta,
        coil_set_spec=coil_set_spec,
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
    )
    return initial_parameters, kernels


def _every_kernel(
    kernels: QfmHostKernels,
    parameters: jax.Array,
) -> tuple[tuple[str, Callable[..., object], tuple[jax.Array, ...]], ...]:
    """Every compiled kernel the module builds, with one set of operands.

    The QFM residual and the three labels take the surface parameters; a
    penalty kernel takes the parameters and its stage target. A kernel missing
    from this tuple is a kernel whose captured constants nothing checks.
    """
    target = parameters[:1].sum()
    return (
        ("qfm", kernels.qfm, (parameters,)),
        *(
            (f"label:{label}", kernels.labels[label], (parameters,))
            for label in QFM_LABELS
        ),
        *(
            (
                f"penalty:{label}",
                kernels.penalty(label=label, constraint_weight=CONSTRAINT_WEIGHT),
                (parameters, target),
            )
            for label in QFM_LABELS
        ),
    )


def _captured_device_arrays(
    closed_jaxpr: extend_core.ClosedJaxpr,
) -> list[tuple[int, ...]]:
    """Shapes of every captured constant that is still on a device.

    A penalty kernel calls the already-jitted QFM and label kernels, and their
    constants stay inside the nested ``pjit`` jaxpr rather than surfacing in
    the outer one, so the walk has to descend: checking only the outer
    ``consts`` reports a penalty kernel as clean while it captures the same
    arrays one level down.
    """
    shapes = [
        tuple(constant.shape)
        for constant in closed_jaxpr.consts
        if isinstance(constant, jax.Array)
    ]
    for equation in closed_jaxpr.jaxpr.eqns:
        for parameter in equation.params.values():
            candidates = (
                parameter if isinstance(parameter, (tuple, list)) else (parameter,)
            )
            for candidate in candidates:
                if isinstance(candidate, extend_core.ClosedJaxpr):
                    shapes.extend(_captured_device_arrays(candidate))
    return shapes


def test_no_qfm_kernel_closes_over_a_device_array() -> None:
    """Lowering may not read a captured constant back to the host."""
    initial_parameters, kernels = _bounded_kernels()
    placed = explicit_device_array(initial_parameters, dtype=np.float64)

    captured_on_device: dict[str, list[tuple[int, ...]]] = {}
    for name, kernel, operands in _every_kernel(kernels, placed):
        on_device = _captured_device_arrays(kernel.trace(*operands).jaxpr)
        if on_device:
            captured_on_device[name] = on_device

    assert not captured_on_device, (
        "these QFM kernels capture device arrays in their closures, which "
        "lowering copies back to the host and the jax-gpu lane refuses: "
        f"{captured_on_device}"
    )


def test_every_qfm_kernel_lowers_under_the_strict_transfer_guard() -> None:
    """The same property as seen by the guard the failing lane actually runs.

    A single-memory CPU backend cannot observe the lowering read-back, so this
    is a GPU-effective pin; the constant-placement test above holds the
    property on every backend.
    """
    initial_parameters, kernels = _bounded_kernels()
    placed = explicit_device_array(initial_parameters, dtype=np.float64)

    for name, kernel, operands in _every_kernel(kernels, placed):
        with disallow_host_transfers():
            value, gradient = kernel(*operands)
        assert np.isfinite(np.asarray(jax.device_get(value))), name
        assert np.all(np.isfinite(np.asarray(jax.device_get(gradient)))), name
