from __future__ import annotations

from unittest import mock

import numpy as np
import simsopt.geo.boozersurface as boozersurface_module
from simsopt.geo.boozersurface import _boozer_exact_newton_observation_context

from .surface_test_helpers import get_boozer_surface


def test_exact_newton_observer_preserves_no_sink_result_and_operation_order() -> None:
    _, boozer_surface = get_boozer_surface(boozer_type="exact", converge=False)
    surface = boozer_surface.surface
    initial_dofs = np.array(surface.get_dofs(), copy=True)
    initial_iota = -0.406
    surface_mask = surface.get_stellsym_mask()
    residual_mask = np.concatenate(
        (
            surface_mask[..., None],
            surface_mask[..., None],
            surface_mask[..., None],
        ),
        axis=2,
    )
    residual_mask[0, 0, 0] = False
    residual_size = residual_mask.size
    state_size = initial_dofs.size + 2

    def run(
        sink: list[dict[str, object]] | None,
    ) -> tuple[dict[str, object], list[str]]:
        operation_order: list[str] = []
        residual_call = [0]

        def residual(_surface, _iota, _G, _field, derivatives=1):
            operation_order.append("residual")
            residual_call[0] += 1
            return (
                np.full(residual_size, float(residual_call[0]), dtype=np.float64),
                np.zeros((residual_size, state_size), dtype=np.float64),
            )

        def solve(_matrix, _rhs):
            operation_order.append("solve")
            return np.ones(state_size, dtype=np.float64)

        surface.set_dofs(initial_dofs)
        boozer_surface.need_to_run_code = True
        with mock.patch.object(
            boozersurface_module,
            "boozer_surface_residual",
            side_effect=residual,
        ), mock.patch.object(
            boozersurface_module.np.linalg, "solve", side_effect=solve
        ):
            if sink is None:
                result = boozer_surface.solve_residual_equation_exactly_newton(
                    tol=1.0e-12,
                    maxiter=2,
                    iota=initial_iota,
                    G=None,
                )
            else:
                with _boozer_exact_newton_observation_context(sink.append):
                    result = boozer_surface.solve_residual_equation_exactly_newton(
                        tol=1.0e-12,
                        maxiter=2,
                        iota=initial_iota,
                        G=None,
                    )
        return result, operation_order

    without_observer, unobserved_order = run(None)
    events: list[dict[str, object]] = []
    with_observer, observed_order = run(events)

    assert unobserved_order == observed_order
    assert unobserved_order == [
        "residual",
        "solve",
        "solve",
        "residual",
        "solve",
        "solve",
        "residual",
        "residual",
    ]
    for field in ("residual", "jacobian", "iota", "G", "success", "iter"):
        np.testing.assert_equal(with_observer[field], without_observer[field])
    assert [event["event"] for event in events] == [
        "assessment",
        "update",
        "assessment",
        "update",
        "terminal",
    ]
