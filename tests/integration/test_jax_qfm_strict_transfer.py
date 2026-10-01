"""Strict-transfer coverage for the exact three-stage QFM workflow.

The mirror is host SciPy over JAX physics, so the boundary is crossed on every
objective evaluation. What must hold is that each crossing is *explicit* and
goes through the repository's owners: no implicit transfer anywhere inside the
solve, and every crossing carrying exactly one value-and-gradient packet.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

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
    solve_qfm_host_scipy_sequence,
    solve_qfm_host_scipy_stage,
)
from simsopt_jax.runtime.host_boundary import (
    disallow_host_transfers,
    host_transfer_audit,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

# One copy of the scale contract, the same table the shipped mirror and the
# parity case read.
BOUNDED_RESOLUTION = QFM_SURFACE_RESOLUTION["bounded"]
BOUNDED_STEPS = 80
TOLERANCE = 1.0e-12


def _bounded_problem() -> tuple[np.ndarray, QfmHostKernels]:
    """The bounded NCSX QFM configuration the parity case also uses."""
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


def test_qfm_sequence_runs_without_any_implicit_host_transfer() -> None:
    """The whole six-call sequence runs under a refusal of implicit transfers."""
    initial_parameters, kernels = _bounded_problem()

    with disallow_host_transfers():
        result = solve_qfm_host_scipy_sequence(
            initial_parameters,
            kernels=kernels,
            max_steps=BOUNDED_STEPS,
            tolerance=TOLERANCE,
            constraint_weight=1.0,
        )

    for label in QFM_LABELS:
        stage = getattr(result, label)
        assert stage.label == label
        assert stage.penalty_optimizer.status == 0
        assert stage.penalty_optimizer.success is True
        assert stage.exact_optimizer.status == 0
        assert stage.exact_optimizer.success is True
        assert stage.penalty_optimizer.nfev >= 1
        assert stage.exact_optimizer.nfev >= 1
        # What a BOUNDED run can prove about its own exact stage: the provider
        # reported its own outcome (above), the endpoint is finite, and the
        # stage moved. No residual ceiling: upstream's ``1_Simple/qfm.py`` has
        # no reduced CI configuration, so this resolution has no official run
        # and no official residual to be measured against. The ceiling derived
        # from the shipped run belongs to the ``native_default`` contract
        # (``native_qfm.NATIVE_DEFAULT_LABEL_RESIDUAL_GROSS_FAILURE_CEILING``).
        assert np.isfinite(stage.exact.label_residual_abs)
        assert np.all(np.isfinite(stage.exact.parameters))
        assert np.any(stage.exact.parameters != stage.initial.parameters)

    assert result.area.exact.qfm_value < result.volume.initial.qfm_value
    assert np.all(np.isfinite(result.area.exact.parameters))
    assert np.all(np.isfinite(result.area.exact.qfm_gradient))


def test_every_audited_crossing_carries_one_value_and_gradient_packet() -> None:
    """Each device-to-host crossing brings back a value and its gradient.

    ``host_transfer_audit`` only sees materializations routed through
    ``host_boundary.host_value``, so a crossing that bypassed the owner, or one
    that fetched a value and its gradient separately, breaks this count.
    """
    initial_parameters, kernels = _bounded_problem()

    with host_transfer_audit() as audit, disallow_host_transfers():
        stage = solve_qfm_host_scipy_stage(
            initial_parameters,
            kernels=kernels,
            label="volume",
            initial_volume=float(kernels.host_label("volume")(initial_parameters)[0]),
            tolerance=TOLERANCE,
            max_steps=BOUNDED_STEPS,
        )

    summaries = audit.summary()
    assert summaries
    total_calls = sum(summary.calls for summary in summaries)
    total_leaves = sum(summary.leaves for summary in summaries)
    assert total_calls >= stage.exact_optimizer.nfev
    assert total_leaves == 2 * total_calls
