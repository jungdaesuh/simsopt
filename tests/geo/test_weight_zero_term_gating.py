"""Zero-weight outer-term gating in the single-stage Boozer objective.

``_traceable_single_stage_outer_term_values`` keeps the three pairwise-distance
penalties out of the traced graph when their configured weight is zero AND
their separation threshold is exactly zero. These tests pin the properties
that make that legal:

1. The gate is numerically invisible -- raw terms, objective value, and
   objective gradient stay bit-identical to the ungated evaluation at the same
   state with the weights still literally ``0.0``.
2. The gate actually removes work -- the ``curve_curve`` penalty, which
   dominates the evaluation, is a constant in the traced program when its
   weight is zero and a live computation when it is not.

They also pin the deliberate exclusion of the ``curvature`` penalty, whose raw
value is nonzero at the shipped zero curvature threshold and is therefore
reported rather than elided.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt.configs import get_data
from simsopt.geo import CurveLength, SurfaceXYZTensorFourier, Volume
from simsopt_jax.core.surface_fourier_indices import stellsym_scatter_indices
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo import surface_objectives

GATED_TERMS = ("curve_curve", "curve_surface", "surface_vessel")


def _hex(value):
    """Return the raw float64 bytes of a scalar, so ties in repr cannot hide."""
    return (
        np.float64(np.asarray(jax.device_get(value), dtype=np.float64)).tobytes().hex()
    )


def _gradient_hex(gradient):
    array = np.asarray(jax.device_get(gradient), dtype=np.float64)
    return array.tobytes().hex()


@pytest.fixture(scope="module")
def single_stage_state():
    """Bounded NCSX single-stage state matching the shipped boozerQA example."""
    base_curves, base_currents, magnetic_axis, nfp, native_field = get_data(
        "ncsx",
        coil_order=3,
        magnetic_axis_order=3,
        points_per_period=8,
    )
    base_currents[0].fix_all()
    field = BiotSavartJAX(native_field.coils)
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))

    mpol = ntor = 2
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, 2 * ntor + 1, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 2 * mpol + 1, endpoint=False),
    )
    surface.fit_to_curve(magnetic_axis, 0.1, flip_theta=True)

    surface_dofs = np.asarray(surface.get_dofs(), dtype=np.float64)
    x_inner = jnp.asarray(
        np.concatenate([surface_dofs, [-0.406], [G0]]), dtype=jnp.float64
    )
    coil_dofs = jnp.asarray(np.asarray(field.x, dtype=np.float64))
    scatter_indices = jnp.asarray(
        np.asarray(stellsym_scatter_indices(mpol, ntor), dtype=np.int32),
        dtype=jnp.int32,
    )
    quadpoints_phi = np.asarray(surface.quadpoints_phi, dtype=np.float64)
    quadpoints_theta = np.asarray(surface.quadpoints_theta, dtype=np.float64)

    qs_resolution = 4
    zero_weight_config = {
        "non_qs_weight": 1.0,
        "residual_weight": 0.0,
        "iota_weight": 1.0,
        "major_radius_weight": 1.0,
        "length_weight": 1.0,
        "curvature_weight": 0.0,
        "curve_curve_weight": 0.0,
        "curve_surface_weight": 0.0,
        "surface_vessel_weight": 0.0,
        "non_qs_quadpoints_phi": np.linspace(
            0.0, 1.0 / nfp, 2 * qs_resolution, endpoint=False
        ),
        "non_qs_quadpoints_theta": np.linspace(
            0.0, 1.0, 2 * qs_resolution, endpoint=False
        ),
        "non_qs_axis": 0,
        "optimized_coil_index": 0,
        "length_coil_indices": (0, 1, 2),
        "length_target": float(sum(CurveLength(c).J() for c in base_curves)),
        "curvature_threshold": 0.0,
        "curvature_p_norm": 2.0,
        "major_radius_target": float(surface.major_radius()),
        "curve_curve_threshold": 0.0,
        "curve_surface_threshold": 0.0,
        "vessel_gamma": np.asarray(surface.gamma(), dtype=np.float64),
        "surface_vessel_threshold": 0.0,
    }
    nonzero_weight_config = dict(zero_weight_config)
    nonzero_weight_config.update(
        curvature_weight=3.0,
        curve_curve_weight=5.0,
        curve_surface_weight=7.0,
        surface_vessel_weight=11.0,
    )

    term_kwargs = {
        "quadpoints_phi": quadpoints_phi,
        "quadpoints_theta": quadpoints_theta,
        "mpol": mpol,
        "ntor": ntor,
        "nfp": nfp,
        "stellsym": True,
        "scatter_indices": scatter_indices,
        "surface_kind": "xyztensorfourier",
        "label_quadpoints_phi": quadpoints_phi,
        "label_quadpoints_theta": quadpoints_theta,
        "label_mpol": mpol,
        "label_ntor": ntor,
        "label_nfp": nfp,
        "label_stellsym": True,
        "label_scatter_indices": scatter_indices,
        "label_surface_kind": "xyztensorfourier",
        "optimize_G": True,
        "weight_inv_modB": True,
        "constraint_weight": None,
        "targetlabel": float(Volume(surface).J()),
        "label_type": "volume",
        "phi_idx": 0,
        "iota_target": -0.406,
        "surface_quadpoints_phi": quadpoints_phi,
        "surface_quadpoints_theta": quadpoints_theta,
        "coil_dof_extraction_spec": field.coil_dof_extraction_spec(),
    }
    return {
        "x_inner": x_inner,
        "coil_dofs": coil_dofs,
        "coil_set_spec": field.coil_set_spec(),
        "zero_weight_config": zero_weight_config,
        "nonzero_weight_config": nonzero_weight_config,
        "term_kwargs": term_kwargs,
    }


def _raw_terms(state, config, coil_dofs=None):
    return surface_objectives._traceable_single_stage_outer_term_values(
        state["x_inner"],
        state["coil_dofs"] if coil_dofs is None else coil_dofs,
        state["coil_set_spec"],
        outer_objective_config=config,
        **state["term_kwargs"],
    )


def _term_program_lines(state, config, term_name):
    """Return the lowered program size of one raw outer term.

    Lowering prunes everything the selected output does not need, so the size
    reports exactly what that single term still costs.
    """

    def term(coil_dofs):
        return _raw_terms(state, config, coil_dofs=coil_dofs)[term_name]

    return len(jax.jit(term).lower(state["coil_dofs"]).as_text().splitlines())


def _objective(state, config):
    def objective(coil_dofs):
        return surface_objectives._traceable_full_single_stage_outer_objective(
            state["x_inner"],
            coil_dofs,
            state["coil_set_spec"],
            outer_objective_config=config,
            **state["term_kwargs"],
        )

    return objective


def test_zero_weight_gate_leaves_every_raw_term_bit_identical(
    single_stage_state, monkeypatch
):
    """Eliding a zero-weight distance term must not perturb any reported term."""
    gated = _raw_terms(single_stage_state, single_stage_state["zero_weight_config"])
    gated_hex = {name: _hex(value) for name, value in gated.items()}

    monkeypatch.setattr(
        surface_objectives,
        "_traceable_single_stage_weight_is_active",
        lambda _weight: True,
    )
    ungated = _raw_terms(single_stage_state, single_stage_state["zero_weight_config"])
    ungated_hex = {name: _hex(value) for name, value in ungated.items()}

    assert gated_hex == ungated_hex, (
        "gating a zero-weight term changed a reported raw outer-term value; the "
        "reporting contract requires the gate to be numerically invisible"
    )
    for name in GATED_TERMS:
        assert gated_hex[name] == "0000000000000000", (
            f"raw[{name}] is not exactly +0.0 under the shipped zero-weight "
            f"configuration, so eliding it would change the reported value"
        )


def test_zero_weight_gate_leaves_objective_value_and_gradient_bit_identical(
    single_stage_state, monkeypatch
):
    """The compiled objective and its gradient must be unchanged by the gate."""
    coil_dofs = single_stage_state["coil_dofs"]
    config = single_stage_state["zero_weight_config"]
    gated_value, gated_gradient = jax.value_and_grad(
        _objective(single_stage_state, config)
    )(coil_dofs)
    gated = (_hex(gated_value), _gradient_hex(gated_gradient))

    monkeypatch.setattr(
        surface_objectives,
        "_traceable_single_stage_weight_is_active",
        lambda _weight: True,
    )
    ungated_value, ungated_gradient = jax.value_and_grad(
        _objective(single_stage_state, config)
    )(coil_dofs)
    ungated = (_hex(ungated_value), _gradient_hex(ungated_gradient))

    assert gated == ungated, (
        "the zero-weight gate changed the objective value or gradient bytes; "
        "the gate is only legal as a bit-identical efficiency fix"
    )


@pytest.mark.parametrize("term_name", GATED_TERMS)
def test_nonzero_weight_keeps_its_own_distance_penalty_traced(
    single_stage_state, term_name
):
    """A term carrying a nonzero weight must stay a live computation."""
    config = dict(single_stage_state["zero_weight_config"])
    config[f"{term_name}_weight"] = 5.0

    live_lines = _term_program_lines(single_stage_state, config, term_name)
    elided_lines = _term_program_lines(
        single_stage_state, single_stage_state["zero_weight_config"], term_name
    )

    assert elided_lines * 2 < live_lines, (
        f"the {term_name} program at weight 5.0 is {live_lines} lines against "
        f"{elided_lines} elided lines; the gate fired for a weight it must "
        "not touch"
    )


@pytest.mark.parametrize("term_name", GATED_TERMS)
def test_zero_weight_nonzero_threshold_keeps_the_computed_raw_value(
    single_stage_state, term_name
):
    """Elision requires a zero threshold, not just a zero weight.

    A zero-weight term with a nonzero separation threshold has a nonzero raw
    penalty; publishing 0.0 for it would silently change the reported
    ``final_<term>_penalty``, so the gate must leave it traced.
    """
    config = dict(single_stage_state["zero_weight_config"])
    config[f"{term_name}_threshold"] = 10.0

    raw = _raw_terms(single_stage_state, config)
    value = float(np.asarray(jax.device_get(raw[term_name])))
    assert value > 0.0, (
        f"raw[{term_name}] reports 0.0 at zero weight and threshold 10.0; the "
        "elision gate fired on a term whose computed penalty is nonzero"
    )


def test_curve_curve_computation_is_absent_from_the_zero_weight_graph(
    single_stage_state,
):
    """The dominant curve-curve penalty must not be traced at zero weight."""

    gated_lines = _term_program_lines(
        single_stage_state, single_stage_state["zero_weight_config"], "curve_curve"
    )
    live_lines = _term_program_lines(
        single_stage_state, single_stage_state["nonzero_weight_config"], "curve_curve"
    )

    assert live_lines > 50, (
        f"the nonzero-weight curve_curve program is only {live_lines} lines, so "
        "this test is not exercising the live pairwise-distance path"
    )
    assert gated_lines * 5 < live_lines, (
        f"the zero-weight curve_curve program is {gated_lines} lines against "
        f"{live_lines} live lines; the penalty is still being computed"
    )


def test_zero_weight_graph_is_cheaper_than_the_nonzero_weight_graph(
    single_stage_state,
):
    """The compiled program that reports the raw terms must shrink measurably."""

    def total_and_raw_terms(config):
        def program(coil_dofs):
            raw = _raw_terms(single_stage_state, config, coil_dofs=coil_dofs)
            weighted = (
                surface_objectives._traceable_weighted_single_stage_outer_term_values(
                    raw, outer_objective_config=config
                )
            )
            total = jnp.asarray(0.0, dtype=jnp.float64)
            for (
                name,
                _weight_key,
            ) in surface_objectives._TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS:
                total = total + weighted[name]
            return total, raw

        cost = (
            jax.jit(program)
            .lower(single_stage_state["coil_dofs"])
            .compile()
            .cost_analysis()
        )
        if isinstance(cost, list):
            cost = cost[0]
        return float(cost["flops"])

    zero_weight_flops = total_and_raw_terms(single_stage_state["zero_weight_config"])
    nonzero_weight_flops = total_and_raw_terms(
        single_stage_state["nonzero_weight_config"]
    )
    assert zero_weight_flops * 2.0 < nonzero_weight_flops, (
        "the zero-weight raw-term program is not measurably smaller "
        f"({zero_weight_flops:.0f} vs {nonzero_weight_flops:.0f} flops), so the "
        "gate is not removing the pairwise-distance work"
    )


def test_curvature_penalty_is_reported_not_elided_at_zero_weight(single_stage_state):
    """The curvature term stays traced: its raw value is nonzero at threshold 0."""
    raw = _raw_terms(single_stage_state, single_stage_state["zero_weight_config"])
    curvature = float(np.asarray(jax.device_get(raw["curvature"])))
    assert curvature > 0.0, (
        "the shipped zero-weight configuration reports a nonzero raw curvature "
        "penalty; gating it would silently change final_curvature_penalty"
    )
