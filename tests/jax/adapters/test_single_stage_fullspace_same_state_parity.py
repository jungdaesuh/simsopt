"""Focused parity tests for the coupled single-stage same-state gate."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.core.specs import (
    make_coil_dof_extraction_spec,
    make_coil_set_dof_extraction_spec,
    make_curve_xyzfourier_spec,
    make_optimizable_dof_map_spec,
    make_surface_xyz_tensor_fourier_spec,
)
from simsopt_jax.core.surface_fourier_indices import stellsym_scatter_indices
from simsopt_jax.objectives.single_stage_fullspace import (
    TERM_LEDGER,
    FullSpaceLayout,
    FullSpaceObjectiveConfig,
    FullSpaceProblem,
    FullSpaceState,
)
from simsopt_jax_adapters.geo.single_stage_fullspace_parity import (
    SameStateEvaluation,
    build_same_state_parity_report,
    evaluate_fullspace_same_state,
)

jax.config.update("jax_enable_x64", True)


def _surface_dofs() -> np.ndarray:
    coefficients = np.zeros((3, 3, 1), dtype=np.float64)
    coefficients[0, 0, 0] = 1.0
    coefficients[0, 1, 0] = 0.2
    coefficients[2, 2, 0] = -0.2
    return coefficients.reshape(-1)[stellsym_scatter_indices(1, 0)]


def _surface_spec(dofs: np.ndarray, nphi: int, ntheta: int):
    return make_surface_xyz_tensor_fourier_spec(
        dofs=dofs,
        quadpoints_phi=np.linspace(0.0, 1.0, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, ntheta, endpoint=False),
        nfp=1,
        stellsym=True,
        mpol=1,
        ntor=0,
    )


def _problem_and_z() -> tuple[FullSpaceProblem, jax.Array]:
    curve_dofs = np.asarray(
        (1.55, 0.0, 0.0, 0.0, 0.0, 0.32, 0.0, 0.32, 0.0),
        dtype=np.float64,
    )
    curve_spec = make_curve_xyzfourier_spec(
        dofs=curve_dofs,
        quadpoints=np.linspace(0.0, 1.0, 16, endpoint=False),
        order=1,
    )
    curve_map = make_optimizable_dof_map_spec(
        template_full_dofs=np.zeros(curve_dofs.size),
        owner_segments=((0, curve_dofs.size, 0, curve_dofs.size),),
        input_mode="full",
        input_start=0,
        input_end=curve_dofs.size,
    )
    fixed_current_map = make_optimizable_dof_map_spec(
        template_full_dofs=np.asarray((1.0e5,)),
        owner_segments=(),
        input_mode="full",
        input_start=0,
        input_end=1,
    )
    extraction = make_coil_set_dof_extraction_spec(
        (
            make_coil_dof_extraction_spec(
                curve=curve_spec,
                curve_map=curve_map,
                current_map=fixed_current_map,
            ),
        )
    )
    surface_dofs = _surface_dofs()
    exact = _surface_spec(surface_dofs, 4, 5)
    layout = FullSpaceLayout(curve_dofs.size, surface_dofs.size)
    problem = FullSpaceProblem(
        coil_dof_extraction=extraction,
        exact_surface_template=exact,
        label_surface_template=_surface_spec(surface_dofs, 6, 7),
        non_qs_surface_template=_surface_spec(surface_dofs, 5, 6),
        exact_mask_indices=jnp.asarray((1, 7, 58), dtype=jnp.int32),
        config=FullSpaceObjectiveConfig(
            iota_target=jnp.asarray(0.37, dtype=jnp.float64),
            major_radius_target=jnp.asarray(0.96, dtype=jnp.float64),
            length_target=jnp.asarray(1.8, dtype=jnp.float64),
            volume_target=jnp.asarray(0.78, dtype=jnp.float64),
            non_qs_weight=jnp.asarray(1.3, dtype=jnp.float64),
            residual_weight=jnp.asarray(0.7, dtype=jnp.float64),
            iota_weight=jnp.asarray(1.1, dtype=jnp.float64),
            major_radius_weight=jnp.asarray(0.9, dtype=jnp.float64),
            length_weight=jnp.asarray(1.2, dtype=jnp.float64),
            non_qs_axis=1,
            weight_inv_modB=False,
            length_coil_indices=(0,),
        ),
        layout=layout,
    )
    z = layout.pack(
        FullSpaceState(
            coil_dofs=jnp.asarray(curve_dofs),
            surface_dofs=jnp.asarray(surface_dofs),
            iota=jnp.asarray(0.31, dtype=jnp.float64),
            G=jnp.asarray(0.85, dtype=jnp.float64),
        )
    )
    return problem, z


def test_real_adapter_and_fullspace_core_match_at_identical_small_state() -> None:
    problem, z = _problem_and_z()

    report = build_same_state_parity_report(z, problem)

    assert report.passed is True
    assert len(report.state_little_endian_sha256) == 64
    assert {comparison.field for comparison in report.comparisons} == {
        "raw_terms.non_qs",
        "raw_terms.residual",
        "raw_terms.iota",
        "raw_terms.major_radius",
        "raw_terms.length",
        "weighted_total",
        "full_boozer_residual",
        "masked_boozer_residual",
        "volume_constraint",
        "observables.iota",
        "observables.G",
        "observables.volume",
        "observables.major_radius",
        "observables.total_length",
        "observables.non_qs_ratio",
        "observables.boozer_residual_scalar",
        "observables.boozer_residual_rms",
    }


def test_report_uses_the_frozen_ledger_tolerance_for_every_field() -> None:
    problem, z = _problem_and_z()
    report = build_same_state_parity_report(
        z,
        problem,
        authoritative_evaluator=evaluate_fullspace_same_state,
    )
    ledger_tolerances = {row.term_id: row.tolerance for row in TERM_LEDGER}
    by_field = {comparison.field: comparison for comparison in report.comparisons}

    assert by_field["raw_terms.non_qs"].tolerance == ledger_tolerances["non_qs"]
    assert (
        by_field["raw_terms.residual"].tolerance
        == ledger_tolerances["boozer_residual_objective"]
    )
    assert (
        by_field["masked_boozer_residual"].tolerance
        == ledger_tolerances["boozer_equality"]
    )
    assert (
        by_field["volume_constraint"].tolerance == ledger_tolerances["volume_equality"]
    )
    assert report.passed is True


@pytest.mark.parametrize(
    ("field", "delta"),
    (
        ("raw_terms", 1.0e-5),
        ("weighted_total", 1.0e-5),
        ("full_boozer_residual", 1.0e-5),
        ("masked_boozer_residual", 1.0e-5),
        ("volume_constraint", 1.0e-5),
        ("observables", 1.0e-5),
    ),
)
def test_report_fails_closed_for_every_output_family(field: str, delta: float) -> None:
    problem, z = _problem_and_z()
    baseline = evaluate_fullspace_same_state(z, problem)

    def changed_authority(
        _z: jax.Array,
        _problem: FullSpaceProblem,
    ) -> SameStateEvaluation:
        if field == "raw_terms":
            return replace(
                baseline,
                raw_terms={
                    **baseline.raw_terms,
                    "length": baseline.raw_terms["length"] + delta,
                },
            )
        if field == "observables":
            return replace(
                baseline,
                observables={
                    **baseline.observables,
                    "boozer_residual_rms": baseline.observables["boozer_residual_rms"]
                    + delta,
                },
            )
        return replace(baseline, **{field: getattr(baseline, field) + delta})

    report = build_same_state_parity_report(
        z,
        problem,
        authoritative_evaluator=changed_authority,
    )

    assert report.passed is False
    assert any(not comparison.passed for comparison in report.comparisons)


def test_report_fails_closed_for_nonfinite_authoritative_output() -> None:
    problem, z = _problem_and_z()
    baseline = evaluate_fullspace_same_state(z, problem)

    report = build_same_state_parity_report(
        z,
        problem,
        authoritative_evaluator=lambda _z, _problem: replace(
            baseline,
            weighted_total=jnp.asarray(jnp.nan, dtype=jnp.float64),
        ),
    )

    comparison = next(
        item for item in report.comparisons if item.field == "weighted_total"
    )
    assert comparison.passed is False
    assert np.isinf(comparison.max_absolute_error)
    assert report.passed is False
