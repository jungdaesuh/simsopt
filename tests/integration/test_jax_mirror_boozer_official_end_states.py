"""Same-state proof for the ``2_Intermediate/boozer.py`` mirror at UPSTREAM's recorded END states.

The native-boozer lanes are judged by upstream's own end-state scatter (``examples/jax/parity/
official_scatter_contracts.py``), which makes every lane-versus-lane end-point comparison informational. That
contract is only sound where the lanes compute upstream's function at upstream's states, so this file evaluates the
JAX lane's penalty residual and its Jacobian -- the quantities its Levenberg-Marquardt stages drive to zero -- at the
area and flux end states the official run recorded (tracked fixture, ``native_default``), against the native
library's evaluation of the same state. At one thread the native kernel IS upstream's: upstream's own extension,
driven by this branch's Python, reproduces the native lane bitwise. The tolerance is the one the workflow test applies
at the initial state (``test_jax_mirror_boozer_parity.py``), not widened.
"""

from __future__ import annotations

import numpy as np
import pytest
from examples.jax.parity.cases.native_boozer import _problem, _scale_configuration
from examples.jax.parity.official_reference import load_official_reference
from simsopt.geo import Area, BoozerSurface, ToroidalFlux
from simsopt_jax.examples.boozer_official import boozer_official_options
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX

import jax
import jax.numpy as jnp

OFFICIAL = load_official_reference("native-boozer")
CONFIGURATION = _scale_configuration("native_default")
CONSTRAINT_WEIGHT = float(CONFIGURATION["constraint_weight"])


def _end_state(stage: str) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(OFFICIAL.array(f"{stage}:surface_dofs"), dtype=np.float64),
            np.asarray(
                [OFFICIAL.scalar(f"{stage}:iota"), OFFICIAL.scalar(f"{stage}:G")],
                dtype=np.float64,
            ),
        )
    )


@pytest.mark.parametrize("stage", ("area", "flux"))
def test_jax_penalty_residual_and_jacobian_equal_native_at_upstreams_end_state(
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    _, native_field, field, surface, _ = _problem(CONFIGURATION)
    # The area stage's target is the fitted start surface's area, as the workflow builds it.
    area = Area(surface)
    assert float(area.J()) == OFFICIAL.scalar("area:target")
    label = area if stage == "area" else ToroidalFlux(surface, field)
    target = float(OFFICIAL.scalar(f"{stage}:target"))
    state = _end_state(stage)

    native_residual, native_jacobian = BoozerSurface(
        native_field, surface, label, target
    )._get_residual_vector_and_jacobian(state, CONSTRAINT_WEIGHT, True, True)

    jax_field = BiotSavartJAX(native_field.coils)
    solver = BoozerSurfaceJAX(
        jax_field,
        surface,
        label,
        target,
        constraint_weight=CONSTRAINT_WEIGHT,
        options=boozer_official_options(
            rough_maxiter=int(CONFIGURATION["jax_bfgs_maxiter"]),
            ls_maxiter=int(CONFIGURATION["jax_ls_maxiter"]),
            tolerance=float(CONFIGURATION["solver_tolerance"]),
        ),
    )
    kernels = solver._get_penalty_kernel_bundle(
        optimize_G=True,
        weight_inv_modB=True,
        constraint_weight=CONSTRAINT_WEIGHT,
    )
    coil_spec = jax_field.coil_set_spec()
    device_state = jnp.asarray(jax.device_put(state))
    jax_residual, jax_jacobian = jax.device_get(
        (
            kernels.residual(device_state, coil_spec),
            kernels.jacobian(device_state, coil_spec),
        )
    )

    np.testing.assert_allclose(
        jax_residual, native_residual, rtol=1.0e-11, atol=1.0e-13
    )
    np.testing.assert_allclose(
        jax_jacobian, native_jacobian, rtol=1.0e-11, atol=1.0e-13
    )
