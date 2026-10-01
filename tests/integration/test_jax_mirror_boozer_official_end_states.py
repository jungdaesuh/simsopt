"""Same-state proof for the ``2_Intermediate/boozer.py`` mirror at UPSTREAM's recorded END states.

The native-boozer lanes are judged by upstream's own end-state scatter (``examples/jax/parity/
official_scatter_contracts.py``), which makes every lane-versus-lane end-point comparison informational. That
contract is only sound where the lanes compute upstream's function at upstream's states (PLAN.md G5), so this file
evaluates the JAX lane's penalty residual and its Jacobian -- the quantities its Levenberg-Marquardt stages drive to
zero -- at the area and flux end states upstream recorded for EVERY branch representative the contract admits, at
EVERY scale it is declared (tracked scatter records; at ``native_default`` the ``k = 0`` representative is the
canonical official capture, bitwise), against the native library's evaluation of the same state. The
representatives are the arbiter's own (lowest-``k`` draw of each branch under the case's manifest routes), so the
proof covers exactly the states a lane can be matched to. At one thread the native kernel IS upstream's: upstream's
own extension, driven by this branch's Python, reproduces the native lane bitwise. The tolerance is the one the
workflow test applies at the initial state (``test_jax_mirror_boozer_parity.py``), not widened.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from examples.jax.manifest_runtime import load_runtime_contract_pair
from examples.jax.parity.arbiter import upstream_branch_representatives
from examples.jax.parity.cases import get_case
from examples.jax.parity.cases.native_boozer import _problem, _scale_configuration
from examples.jax.parity.official_reference import (
    UpstreamScatterRun,
    load_upstream_scatter,
)
from simsopt.geo import Area, BoozerSurface, ToroidalFlux
from simsopt_jax.examples import ExecutionScale
from simsopt_jax.examples.boozer_official import boozer_official_options
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX

import jax
import jax.numpy as jnp

CASE_ID = "native-boozer"
SCALES: tuple[ExecutionScale, ...] = ("bounded", "native_default")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _admitted_representatives() -> tuple[tuple[ExecutionScale, int], ...]:
    """Every (scale, k) the declared end-state sets can match a lane to."""
    pair = load_runtime_contract_pair(
        REPO_ROOT / "examples/jax/manifest.json",
        REPO_ROOT / "examples/jax/parity_manifest.json",
        repo_root=REPO_ROOT,
    )
    relationship = next(
        item for item in pair.parity.relationships if item.case_id == CASE_ID
    )
    admitted: list[tuple[ExecutionScale, int]] = []
    for scale in SCALES:
        end_states = get_case(CASE_ID).end_states(scale)
        if end_states is None:
            continue
        routes = relationship.resolve_scale(scale).comparison_routes
        admitted.extend(
            (scale, state.k)
            for state in upstream_branch_representatives(end_states, routes)
        )
    return tuple(admitted)


ADMITTED = _admitted_representatives()


def _end_state(run: UpstreamScatterRun, stage: str) -> np.ndarray:
    return np.concatenate(
        (
            run.value(f"{stage}:surface_dofs"),
            np.asarray(
                [run.value(f"{stage}:iota"), run.value(f"{stage}:G")],
                dtype=np.float64,
            ),
        )
    )


def test_every_declared_scale_is_covered() -> None:
    """The proof below runs at every scale that declares an end-state set, and only there."""
    assert {scale for scale, _k in ADMITTED} == {
        scale for scale in SCALES if get_case(CASE_ID).end_states(scale) is not None
    }


@pytest.mark.parametrize("stage", ("area", "flux"))
@pytest.mark.parametrize(
    ("scale", "k"), ADMITTED, ids=[f"{scale}-k{k}" for scale, k in ADMITTED]
)
def test_jax_penalty_residual_and_jacobian_equal_native_at_upstreams_end_state(
    scale: ExecutionScale,
    k: int,
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    configuration = _scale_configuration(scale)
    constraint_weight = float(configuration["constraint_weight"])
    run = load_upstream_scatter(CASE_ID, scale).run(k)
    _, native_field, field, surface, _ = _problem(configuration)
    # The area stage's target is the fitted start surface's area, as the workflow builds it.
    area = Area(surface)
    assert float(area.J()) == float(run.value("area:target"))
    label = area if stage == "area" else ToroidalFlux(surface, field)
    target = float(run.value(f"{stage}:target"))
    state = _end_state(run, stage)

    native_residual, native_jacobian = BoozerSurface(
        native_field, surface, label, target
    )._get_residual_vector_and_jacobian(state, constraint_weight, True, True)

    jax_field = BiotSavartJAX(native_field.coils)
    solver = BoozerSurfaceJAX(
        jax_field,
        surface,
        label,
        target,
        constraint_weight=constraint_weight,
        options=boozer_official_options(
            rough_maxiter=int(configuration["jax_bfgs_maxiter"]),
            ls_maxiter=int(configuration["jax_ls_maxiter"]),
            tolerance=float(configuration["solver_tolerance"]),
        ),
    )
    kernels = solver._get_penalty_kernel_bundle(
        optimize_G=True,
        weight_inv_modB=True,
        constraint_weight=constraint_weight,
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
