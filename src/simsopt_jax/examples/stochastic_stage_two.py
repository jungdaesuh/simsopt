"""Single source of truth for the stochastic stage-two example scales."""

from __future__ import annotations

from dataclasses import dataclass

from .execution import ExecutionScale


@dataclass(frozen=True, slots=True)
class StochasticStageTwoConfiguration:
    """Immutable scientific and numerical configuration for one execution scale."""

    surface_nphi: int
    surface_ntheta: int
    curve_order: int
    curve_quadrature: int
    num_base_curves: int
    major_radius: float
    minor_radius: float
    initial_current: float
    length_weight: float
    curve_curve_threshold: float
    curve_curve_weight: float
    curvature_threshold: float
    curvature_weight: float
    mean_squared_curvature_threshold: float
    mean_squared_curvature_weight: float
    arclength_variation_weight: float
    perturbation_sigma: float
    perturbation_length_scale: float
    training_sample_count: int
    out_of_sample_count: int
    training_seed: int
    out_of_sample_seed: int
    max_steps: int
    rtol: float
    atol: float


def stochastic_stage_two_configuration(
    scale: ExecutionScale,
) -> StochasticStageTwoConfiguration:
    """Return the canonical stochastic stage-two configuration."""
    native_scale = scale == "native_default"
    return StochasticStageTwoConfiguration(
        surface_nphi=64 if native_scale else 4,
        surface_ntheta=16 if native_scale else 4,
        curve_order=24 if native_scale else 2,
        curve_quadrature=360 if native_scale else 16,
        num_base_curves=4,
        major_radius=1.0,
        minor_radius=0.5,
        initial_current=1.0e5,
        length_weight=1.0e-6,
        curve_curve_threshold=0.1,
        curve_curve_weight=10.0,
        curvature_threshold=5.0,
        curvature_weight=1.0e-6,
        mean_squared_curvature_threshold=5.0,
        mean_squared_curvature_weight=1.0e-6,
        arclength_variation_weight=1.0e-2,
        perturbation_sigma=1.0e-3,
        perturbation_length_scale=0.5,
        training_sample_count=16 if native_scale else 2,
        out_of_sample_count=256 if native_scale else 4,
        training_seed=0,
        out_of_sample_seed=1,
        max_steps=400 if native_scale else 20,
        rtol=1.0e-15,
        atol=1.0e-8,
    )
