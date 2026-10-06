"""BoozerLS Newton polish options of ``BoozerSurfaceJAX``.

``newton_assembly`` selects the polish ``run_code`` runs (``"ad"``: basis
HVPs of the AD penalty, the default; ``"analytic"``: native-order dense-LU
Newton over the analytic Hessian).  ``newton_divergence_factor`` is the
analytic Newton's entry-residual blow-up guard (default 1e3, ``None``
disables it; the AD polish ignores it).  The analytic Newton refuses
settings it cannot honour instead of silently running something else.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import math

import numpy as np
import pytest
import simsopt_jax_adapters.geo.boozer_surface as boozer_surface_module
from simsopt.configs import get_data
from simsopt.geo import SurfaceXYZTensorFourier, Volume
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import (
    BOOZER_LS_DERIVATIVE_ASSEMBLIES,
    BoozerSurfaceJAX,
)

CONSTRAINT_WEIGHT = 11.1232
IOTA = -0.37


def _boozer_ls(options: dict[str, object] | None = None):
    """A low-order NCSX BoozerLS problem: ``(boozer_surface, G)``."""
    _, currents, axis, nfp, field = get_data(
        "ncsx", coil_order=3, magnetic_axis_order=3, points_per_period=8
    )
    surface = SurfaceXYZTensorFourier(
        mpol=1,
        ntor=1,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(0, 1 / nfp, 3, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 3, endpoint=False),
    )
    surface.fit_to_curve(axis, 0.1, flip_theta=True)
    label = Volume(surface)
    boozer_surface = BoozerSurfaceJAX(
        BiotSavartJAX(field.coils),
        surface,
        label,
        label.J() * 1.01,
        constraint_weight=CONSTRAINT_WEIGHT,
        options={"verbose": False, **(options or {})},
    )
    G = 4e-7 * np.pi * nfp * sum(abs(current.get_value()) for current in currents)
    return boozer_surface, G


def test_the_boozer_ls_derivative_assemblies_are_ad_and_analytic() -> None:
    assert BOOZER_LS_DERIVATIVE_ASSEMBLIES == ("ad", "analytic")


def test_default_options_select_the_ad_polish_with_the_default_guard() -> None:
    boozer_surface, _G = _boozer_ls()

    assert boozer_surface.options["newton_assembly"] == "ad"
    assert boozer_surface.options["newton_divergence_factor"] == 1e3


@pytest.mark.parametrize("newton_assembly", BOOZER_LS_DERIVATIVE_ASSEMBLIES)
def test_default_options_round_trip_through_the_constructor(newton_assembly) -> None:
    first, _G = _boozer_ls({"newton_assembly": newton_assembly})

    second = BoozerSurfaceJAX(
        first.biotsavart,
        first.surface,
        first.label,
        first.targetlabel,
        constraint_weight=first.constraint_weight,
        options=dict(first.options),
    )

    assert second.options == first.options
    assert first.options["newton_assembly"] == newton_assembly
    assert first.options["newton_divergence_factor"] == 1e3, (
        "both assemblies keep the analytic guard's default; the AD polish ignores it"
    )


def test_an_unknown_newton_assembly_is_rejected() -> None:
    with pytest.raises(
        ValueError, match="newton_assembly must be one of: ad, analytic"
    ):
        _boozer_ls({"newton_assembly": "symbolic"})


@pytest.mark.parametrize(
    "factor",
    [True, False, 0, 0.0, -1.0, math.inf, math.nan, "1e3"],
    ids=["true", "false", "int-zero", "zero", "negative", "inf", "nan", "string"],
)
def test_a_divergence_factor_that_is_not_none_or_positive_finite_is_rejected(
    factor,
) -> None:
    """A bool is not a factor, although ``True == 1`` in Python."""
    with pytest.raises(
        ValueError,
        match="newton_divergence_factor must be None or a positive finite number",
    ):
        _boozer_ls({"newton_divergence_factor": factor})


@pytest.mark.parametrize("factor", [None, 5.0, 10])
def test_a_divergence_factor_that_is_none_or_positive_finite_is_kept(factor) -> None:
    boozer_surface, _G = _boozer_ls({"newton_divergence_factor": factor})

    assert boozer_surface.options["newton_divergence_factor"] == factor


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {"newton_linear_solver": "operator_gmres"},
            "cannot honour newton_linear_solver='operator_gmres'",
        ),
        (
            {"materialize_dense_linearization": False},
            "cannot honour materialize_dense_linearization=False",
        ),
    ],
)
def test_the_analytic_assembly_rejects_settings_it_cannot_honour(
    options, message
) -> None:
    """The analytic Newton always factors its materialized Hessian with dense LU."""
    with pytest.raises(ValueError, match=message):
        _boozer_ls({"newton_assembly": "analytic", **options})

    boozer_surface, _G = _boozer_ls({"newton_assembly": "ad", **options})
    for name, value in options.items():
        assert boozer_surface.options[name] == value, (
            "the AD polish honours these settings and keeps them"
        )


def test_the_analytic_newton_refuses_a_hessian_over_the_dense_budget() -> None:
    boozer_surface, G = _boozer_ls(
        {"newton_assembly": "analytic", "max_dense_linearization_bytes": 1024}
    )
    size = len(boozer_surface.surface.get_dofs()) + 2
    assert size * size * 8 > 1024, "the fixture's Hessian must exceed the budget"

    with pytest.raises(
        ValueError,
        match=rf"cannot honour max_dense_linearization_bytes=1024 \(the {size}x{size}",
    ):
        boozer_surface._minimize_boozer_penalty_constraints_newton_analytic(
            constraint_weight=CONSTRAINT_WEIGHT, iota=IOTA, G=G
        )


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, 1e3),
        ({"newton_divergence_factor": None}, None),
        ({"newton_divergence_factor": 5.0}, 5.0),
    ],
    ids=["default", "disabled", "configured"],
)
def test_the_analytic_newton_walk_is_built_with_the_configured_guard(
    monkeypatch, options, expected
) -> None:
    guards = []
    kernel = boozer_surface_module.newton_ls_native_dense

    def recorded_kernel(*args, divergence_factor, **kwargs):
        guards.append(divergence_factor)
        return kernel(*args, divergence_factor=divergence_factor, **kwargs)

    monkeypatch.setattr(
        boozer_surface_module, "newton_ls_native_dense", recorded_kernel
    )
    boozer_surface, G = _boozer_ls({"newton_assembly": "analytic", **options})

    result = boozer_surface._minimize_boozer_penalty_constraints_newton_analytic(
        constraint_weight=CONSTRAINT_WEIGHT, iota=IOTA, G=G
    )

    assert guards == [expected], (
        "the Newton walk must be traced with the option's guard"
    )
    assert np.isfinite(float(result["fun"]))


def _count_newton_routes(boozer_surface, monkeypatch) -> dict[str, int]:
    """Count run_code's calls into each Newton polish (each call still runs)."""
    calls = {"ad": 0, "analytic": 0}
    for name, method in (
        ("ad", "minimize_boozer_penalty_constraints_newton"),
        ("analytic", "_minimize_boozer_penalty_constraints_newton_analytic"),
    ):
        original = getattr(boozer_surface, method)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(boozer_surface, method, counted)
    return calls


@pytest.mark.parametrize("newton_assembly", BOOZER_LS_DERIVATIVE_ASSEMBLIES)
def test_run_code_polishes_with_the_selected_assembly(
    monkeypatch, newton_assembly
) -> None:
    boozer_surface, G = _boozer_ls(
        {"newton_assembly": newton_assembly, "bfgs_maxiter": 20}
    )
    calls = _count_newton_routes(boozer_surface, monkeypatch)

    result = boozer_surface.run_code(IOTA, G)

    assert calls == {
        name: int(name == newton_assembly) for name in BOOZER_LS_DERIVATIVE_ASSEMBLIES
    }
    assert np.isfinite(float(result["fun"]))
