"""Exact-Boozer single-stage outer evaluation with the analytic dense inner Newton.

One outer evaluation mirrors the native reference
(``examples/3_Advanced/single_stage_boozer_vacuum_optimization.py``): the inner
exact Newton warm-starts from the last *successful* surface state, keeps
native's tolerance, iteration cap and rollback rule, and an unsuccessful inner
solve reports the native sentinel value with the gradient evaluated at the
*returned* (failed) iterate. Native computes value and gradient at that returned
iterate and only then restores the pre-evaluation state, so the failed iterate
is never carried into the next warm start.
The inner solve is the on-device native-order C2 runner with the analytic masked
residual/Jacobian; the outer gradient is the implicit-function derivative built
from that returned-state Jacobian:

    J^T lambda = dJ/dx,        dJ/dc = partial_c J - (dF/dc)^T lambda,

with ``F`` the masked exact residual plus the label row and ``J = dF/dx``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import lu_factor, lu_solve
from numpy.typing import NDArray
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.core._math_utils import as_jax_float64
from simsopt_jax.core.field import (
    coil_set_spec_from_dof_extraction_spec,
    grouped_field_data_from_spec,
)
from simsopt_jax.core.specs import host_resident_spec
from simsopt_jax.runtime.host_boundary import block_until_ready, host_value

from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import (
    BoozerSurfaceJAX,
    _grouped_coil_set_static_signature,
    _label_value_from_surface_dofs,
)
from simsopt_jax_adapters.geo.single_stage_host_construction import (
    host_analytic_geometry_origin_and_basis,
    host_coil_currents,
    host_grouped_coil_set_spec,
)
from simsopt_jax_adapters.geo.surface_objectives_traceable import (
    _build_traceable_objective_cache_state,
    _evaluate_traceable_total_objective,
)

__all__ = [
    "INNER_FAILURE_VALUE",
    "ExactAnalyticEvaluation",
    "ExactAnalyticSingleStage",
    "HostConstructionBoozerSurfaceJAX",
]

# The native example returns this objective value when the inner exact Newton
# solve does not converge, so the outer line search backs off.
INNER_FAILURE_VALUE = 1.0e3


class HostConstructionBoozerSurfaceJAX(BoozerSurfaceJAX):
    """Exact Boozer adapter whose construction-time constants stay on the host.

    Coil groups and the affine surface basis are baked with NumPy. The compiled
    evaluate program still captures the same host constants as the JAX bake.
    """

    def _refresh_coil_data(self):
        coil_set_spec = host_grouped_coil_set_spec(self.biotsavart)
        coil_set_static_signature = _grouped_coil_set_static_signature(coil_set_spec)
        previous_coil_set_static_signature = self._coil_set_static_signature
        self.coil_set_spec = coil_set_spec
        self.coil_groups = list(grouped_field_data_from_spec(self.coil_set_spec))
        self.coil_currents = host_coil_currents(self.coil_set_spec)
        self._coil_set_static_signature = coil_set_static_signature
        self._reference_penalty_objective_cache.clear()
        self._reference_penalty_value_and_grad_cache.clear()
        self._reference_penalty_residual_cache.clear()
        self._reference_exact_residual_cache.clear()
        if (
            previous_coil_set_static_signature is not None
            and previous_coil_set_static_signature != coil_set_static_signature
        ):
            self._kernel_bundle_cache.clear()

    def _make_analytic_geometry_terms(self):
        surface_args = self._traceable_surface_runtime_args()
        origin, basis = host_analytic_geometry_origin_and_basis(
            n_dofs=int(np.asarray(self.surface.get_dofs()).size),
            quadpoints_phi=surface_args["quadpoints_phi"],
            quadpoints_theta=surface_args["quadpoints_theta"],
            mpol=surface_args["mpol"],
            ntor=surface_args["ntor"],
            nfp=surface_args["nfp"],
            stellsym=surface_args["stellsym"],
            scatter_indices=surface_args["scatter_indices"],
            surface_kind=surface_args["surface_kind"],
            clamped_dims=surface_args["clamped_dims"],
        )

        def geometry_from_dofs(sdofs):
            return jax.tree.map(
                lambda offset, coefficients: (
                    offset + jnp.einsum("...s,s->...", coefficients, sdofs)
                ),
                origin,
                basis,
            )

        label_args = {
            key: value
            for key, value in surface_args.items()
            if key.startswith("label_") or key == "phi_idx"
        }

        def label_value(sdofs, coil_set_spec):
            return _label_value_from_surface_dofs(
                sdofs, coil_set_spec=coil_set_spec, **label_args
            )

        return geometry_from_dofs, basis, label_value


@dataclass(frozen=True)
class ExactAnalyticEvaluation:
    """One outer evaluation: reported value, gradient and inner-solve facts."""

    value: float
    gradient: NDArray[np.float64]
    inner_success: bool
    inner_iterations: int
    seconds: float


class ExactAnalyticSingleStage:
    """Stateful outer evaluator; the state is the last *successful* inner solution.

    ``boozer_surface`` must be an exact ``BoozerSurfaceJAX`` sharing ``field``.
    Construction runs the initial inner solve from ``(iota, G)`` at the surface's
    current coefficients, publishes it as the surface's solved state and freezes
    ``iota_target`` at that solution, as the native example does.
    ``outer_objective_config`` is called after that solve, so targets it derives
    from the surface (native takes the major-radius target from the solved
    surface) see the solved coefficients.
    """

    def __init__(
        self,
        boozer_surface: BoozerSurfaceJAX,
        field: BiotSavartJAX,
        *,
        iota: float,
        G: float,
        outer_objective_config: Callable[[], Mapping[str, object]],
    ) -> None:
        if boozer_surface.boozer_type != "exact":
            raise ValueError("ExactAnalyticSingleStage requires boozer_type='exact'.")
        route = boozer_surface._make_run_code_traceable_exact_benchmark_variant(
            "C2", analytic=True, condition_estimate=False
        )
        # Every program below captures the coil graph in its closure instead of
        # taking it as an argument, so the frozen reconstruction template is
        # materialized on the host once here; see
        # ``simsopt_jax.core.specs.host_resident_spec``. ``BiotSavartJAX``
        # keeps its own copy device-resident for the callers that do pass it
        # as a program argument.
        coil_extraction_spec = host_resident_spec(field.coil_dof_extraction_spec())

        def coil_set_spec_from_dofs(coil_dofs):
            return coil_set_spec_from_dof_extraction_spec(
                coil_extraction_spec, as_jax_float64(coil_dofs)
            )

        def solve(coil_dofs, x_inner):
            # The coil spec is built inside the compiled program: eager
            # construction from host dofs costs ~0.1 s of small dispatches per
            # evaluation, several times the solve itself.
            return route.compiled_kernel(
                coil_set_spec_from_dofs(coil_dofs),
                None,
                x_inner[:-2],
                x_inner[-2],
                x_inner[-1],
            )

        # The seed coils and the seed decision vector are host data. Assemble
        # them with NumPy and cross to the device once, explicitly, so the
        # construction path stays clean under ``jax.transfer_guard("disallow")``
        # -- an eager ``jnp`` constructor over host values crosses implicitly.
        seed_coil_dofs = explicit_device_array(
            np.asarray(field.x, dtype=np.float64), dtype=np.float64
        )
        seed_inner = explicit_device_array(
            np.concatenate(
                (
                    np.asarray(boozer_surface.surface.get_dofs(), dtype=np.float64),
                    np.asarray((iota, G), dtype=np.float64),
                )
            ),
            dtype=np.float64,
        )
        initial = block_until_ready(jax.jit(solve)(seed_coil_dofs, seed_inner))
        # ``bool``/``float``/``int`` of a device array is an implicit read back;
        # the three fields the host keeps cross once, explicitly, through the
        # host-boundary owner.
        initial_success, initial_iota, initial_iterations = host_value(
            (initial["success"], initial["iota"], initial["nit"])
        )
        if not bool(initial_success):
            raise RuntimeError("The initial exact Boozer solve did not converge.")
        boozer_surface.install_traceable_solved_runtime_state(
            route.project_result(initial)
        )
        self.iota_target = float(initial_iota)
        self.initial_inner_iterations = int(initial_iterations)
        cache_state = _build_traceable_objective_cache_state(
            boozer_surface,
            field,
            self.iota_target,
            outer_objective_config=outer_objective_config(),
            require_ondevice_inner=False,
        )
        self._objective_cache_state = cache_state
        objective_kwargs = cache_state["objective_kwargs"]
        value_jacobian = boozer_surface._make_analytic_exact_value_jacobian(
            boozer_surface.options["weight_inv_modB"]
        )

        def objective(x_inner, coil_dofs):
            return _evaluate_traceable_total_objective(
                x_inner,
                coil_dofs,
                coil_set_spec_from_dofs(coil_dofs),
                objective_kwargs,
            )

        def value_and_gradient(coil_dofs, x_solved, jacobian, success):
            value, objective_pullback = jax.vjp(objective, x_solved, coil_dofs)
            dJ_dx, dJ_dc = objective_pullback(jnp.ones((), dtype=value.dtype))
            adjoint = lu_solve(lu_factor(jacobian), dJ_dx, trans=1)

            def residual_of_coils(current_coil_dofs):
                return value_jacobian(
                    x_solved, coil_set_spec_from_dofs(current_coil_dofs)
                )[0]

            _, residual_pullback = jax.vjp(residual_of_coils, coil_dofs)
            (adjoint_dF_dc,) = residual_pullback(adjoint)
            reported = jnp.where(
                success, value, jnp.asarray(INNER_FAILURE_VALUE, dtype=value.dtype)
            )
            return reported, dJ_dc - adjoint_dF_dc

        def evaluate(coil_dofs, x_inner):
            solved = solve(coil_dofs, x_inner)
            value, gradient = value_and_gradient(
                coil_dofs, solved["x"], solved["jacobian"], solved["success"]
            )
            return solved["x"], solved["success"], solved["nit"], value, gradient

        # One program per evaluation: the solve and the implicit gradient are
        # dispatched together, with a single device-to-host transfer at the end.
        self._evaluate_kernel = jax.jit(evaluate)
        # Every later inner state is a committed output of that program; the
        # initial one is committed explicitly so the first call compiles the
        # same executable instead of a second, uncommitted-input variant.
        self._x = explicit_device_array(
            initial["x"], dtype=initial["x"].dtype, reference=initial["x"]
        )
        self.coil_dofs = np.asarray(field.x, dtype=np.float64)

    @property
    def x_inner(self) -> jax.Array:
        """Warm start for the next solve: ``[surface_dofs, iota, G]`` (device array).

        This is the state returned by the last inner solve that succeeded. A
        failed solve's returned iterate is used for that evaluation's value and
        gradient but never becomes the warm start.
        """
        return self._x

    def evaluate(self, coil_dofs: object) -> ExactAnalyticEvaluation:
        started = time.perf_counter()
        coil = jnp.asarray(np.asarray(coil_dofs, dtype=np.float64).reshape(-1))
        x_returned, success, iterations, value, gradient = self._evaluate_kernel(
            coil, self._x
        )
        success_host, iterations_host, value_host, gradient_host = host_value(
            (success, iterations, value, gradient)
        )
        # Value and gradient above are always those of ``x_returned``, the
        # iterate the inner solve actually returned, exactly as native computes
        # them before restoring. The restore is a separate step and applies to
        # the warm start only: on failure the pre-evaluation state is kept, so
        # ``self._x`` stays at the last successful solution and the failed
        # iterate is discarded.
        if bool(success_host):
            self._x = x_returned
        return ExactAnalyticEvaluation(
            value=float(value_host),
            gradient=np.asarray(gradient_host, dtype=np.float64),
            inner_success=bool(success_host),
            inner_iterations=int(iterations_host),
            seconds=time.perf_counter() - started,
        )

    def scipy_value_and_gradient(
        self, parameters: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """``scipy.optimize.minimize(jac=True)`` callable with the native policy."""
        evaluation = self.evaluate(parameters)
        return evaluation.value, evaluation.gradient
