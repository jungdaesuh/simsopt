from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.single_stage_optimization.banana_opt.jax_banana_specs import (
    build_banana_local_objective_spec_from_biotsavart,
    build_banana_local_objective_spec_from_coils,
)
from examples.single_stage_optimization.banana_opt.jax_banana_types import (
    BANANA_IDX,
    DEFAULT_BANANA_DOFS,
    DEFAULT_PROXY_RZ,
    HARDWARE_LIMITS,
    HBT_BANANA_WS,
    KA_TO_A,
    N_BANANA,
    N_PROXY,
    N_TF,
    N_VF,
    SingleStageWeights,
    Stage2Weights,
)
from simsopt_jax.core.banana import (
    banana_geometry_from_dofs,
    banana_local_terms,
    banana_local_value,
)
from simsopt_jax.core.specs import make_curve_xyzfourier_spec
from simsopt_jax_adapters.objectives.stage2_target import (
    stage2_target_optimizer_state_from_dofs,
)


jax.config.update("jax_enable_x64", True)


def _stage2_weights_with_single_stage_geometry() -> Stage2Weights:
    single_stage_weights = SingleStageWeights()
    return replace(
        Stage2Weights(),
        sqflux=0.0,
        length=single_stage_weights.length,
        ccdist=single_stage_weights.ccdist,
        csdist=single_stage_weights.csdist,
        curvature=single_stage_weights.curvature,
        poloidal=single_stage_weights.poloidal,
        width=single_stage_weights.width,
        selfint=single_stage_weights.selfint,
        currents=single_stage_weights.currents,
    )


@dataclass
class _HostCurve:
    dofs: np.ndarray
    quadpoints: np.ndarray
    order: int = 1

    def get_dofs(self) -> np.ndarray:
        return self.dofs.copy()

    def to_spec(self):
        return make_curve_xyzfourier_spec(
            dofs=self.dofs,
            quadpoints=self.quadpoints,
            order=self.order,
        )


@dataclass
class _HostCurrent:
    full_value: float
    free_value: float | None

    @property
    def local_x(self) -> np.ndarray:
        if self.free_value is None:
            return np.asarray((), dtype=np.float64)
        return np.asarray([self.free_value], dtype=np.float64)

    @property
    def local_full_x(self) -> np.ndarray:
        return np.asarray([self.full_value], dtype=np.float64)

    def get_value(self) -> float:
        return self.full_value


@dataclass
class _BadVectorCurrent:
    @property
    def local_x(self) -> np.ndarray:
        return np.asarray([1.0, 2.0], dtype=np.float64)

    @property
    def local_full_x(self) -> np.ndarray:
        return np.asarray([1.0], dtype=np.float64)

    def get_value(self) -> float:
        return 1.0


@dataclass
class _ScaledCurrent:
    current_to_scale: object
    scale: float

    def get_value(self) -> float:
        return float(self.scale) * float(self.current_to_scale.get_value())


@dataclass
class _RotatedCurve:
    curve: object
    rotmat: np.ndarray


@dataclass
class _HostCoil:
    curve: object
    current: object


@dataclass
class _HostBiotSavart:
    coils: list[_HostCoil]


@dataclass
class _HostSurface:
    surface_gamma: np.ndarray
    surface_normal: np.ndarray

    def gamma(self) -> np.ndarray:
        return self.surface_gamma.copy()

    def normal(self) -> np.ndarray:
        return self.surface_normal.copy()


def _make_curve() -> _HostCurve:
    return _HostCurve(
        dofs=np.array(
            [0.0, 0.7, 0.1, 0.0, -0.2, 0.9, 0.1, 0.2, -0.1],
            dtype=np.float64,
        ),
        quadpoints=np.linspace(0.0, 1.0, 8, endpoint=False, dtype=np.float64),
    )


def _make_banana_coils(
    *,
    curve: _HostCurve | None = None,
    current: object | None = None,
) -> list[_HostCoil]:
    base_curve = _make_curve() if curve is None else curve
    base_current = _HostCurrent(full_value=1.25, free_value=1.25)
    current_owner = base_current if current is None else current
    rotmat = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return [
        _HostCoil(base_curve, _ScaledCurrent(current_owner, 16.0 * KA_TO_A)),
        _HostCoil(
            _RotatedCurve(base_curve, rotmat),
            _ScaledCurrent(current_owner, -16.0 * KA_TO_A),
        ),
    ]


def test_build_banana_local_spec_from_coils_preserves_current_first_layout():
    banana_coils = _make_banana_coils()
    spec, decision_vector = build_banana_local_objective_spec_from_coils(
        banana_coils,
        stage="stage2",
        weights=Stage2Weights(),
        include_current_penalty=True,
        include_min_length=True,
        include_width=True,
    )

    assert spec.stage == "stage2"
    assert spec.system.decision.tf_current_dof_count == 0
    assert spec.system.decision.current_dof_count == 1
    assert spec.system.decision.curve_dof_count == banana_coils[0].curve.get_dofs().size
    np.testing.assert_allclose(decision_vector[0], np.array(1.25))
    np.testing.assert_allclose(
        decision_vector[1:],
        banana_coils[0].curve.get_dofs(),
    )
    assert spec.length_max_threshold == HARDWARE_LIMITS.max_length
    assert spec.length_max_mode == "max"
    assert spec.length_min_threshold == 0.5 * HARDWARE_LIMITS.max_length
    assert spec.length_min_mode == "min"
    assert spec.width_major_radius == HBT_BANANA_WS.major_radius
    assert spec.width_max_mode == "max"
    assert spec.width_min_mode == "min"
    assert spec.coil_coil_distance_minimum == HARDWARE_LIMITS.min_ccdist
    assert spec.tf_current_max_threshold == 80.0 * KA_TO_A
    assert spec.tf_current_max_mode == "max"
    assert spec.banana_current_max_threshold == 16.0 * KA_TO_A
    assert spec.banana_current_max_mode == "max"
    assert spec.length_weight == Stage2Weights().length
    assert spec.coil_coil_distance_weight == Stage2Weights().ccdist
    assert spec.tf_current_weight == Stage2Weights().currents
    assert spec.banana_current_weight == Stage2Weights().currents

    geometry = banana_geometry_from_dofs(
        spec.system,
        jnp.asarray(decision_vector, dtype=jnp.float64),
    )
    np.testing.assert_allclose(geometry.currents, np.array([20000.0, -20000.0]))
    np.testing.assert_allclose(geometry.tf_current, np.array(0.0))
    terms = banana_local_terms(
        spec,
        jnp.asarray(decision_vector, dtype=jnp.float64),
    )
    np.testing.assert_allclose(
        terms.banana_current_max,
        np.array(0.5 * (20000.0 - 16000.0) ** 2),
    )


def test_build_banana_local_spec_from_coils_freezes_surface_samples():
    surface = _HostSurface(
        surface_gamma=np.array(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            dtype=np.float64,
        ),
        surface_normal=np.array(
            [[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]],
            dtype=np.float64,
        ),
    )

    spec, _decision_vector = build_banana_local_objective_spec_from_coils(
        _make_banana_coils(),
        stage="stage2",
        weights=Stage2Weights(),
        surface=surface,
        include_current_penalty=True,
        include_min_length=True,
        include_width=True,
    )

    assert spec.include_coil_surface_distance is True
    assert spec.coil_surface_distance_minimum == HARDWARE_LIMITS.min_csdist
    assert spec.coil_surface_distance_weight == Stage2Weights().csdist
    np.testing.assert_allclose(
        spec.surface_gamma,
        surface.surface_gamma.reshape((-1, 3)),
    )
    np.testing.assert_allclose(
        spec.surface_normal,
        surface.surface_normal.reshape((-1, 3)),
    )


def test_build_banana_local_spec_from_coils_can_explicitly_disable_surface_term():
    surface = _HostSurface(
        surface_gamma=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        surface_normal=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
    )

    spec, _decision_vector = build_banana_local_objective_spec_from_coils(
        _make_banana_coils(),
        stage="stage2",
        weights=Stage2Weights(),
        surface=surface,
        include_current_penalty=True,
        include_min_length=True,
        include_width=True,
        include_coil_surface_distance=False,
    )

    assert spec.include_coil_surface_distance is False


def test_build_banana_local_spec_from_coils_requires_surface_for_coil_surface_term():
    with pytest.raises(ValueError, match="require a surface"):
        build_banana_local_objective_spec_from_coils(
            _make_banana_coils(),
            stage="stage2",
            weights=Stage2Weights(),
            include_current_penalty=True,
            include_min_length=True,
            include_width=True,
            include_coil_surface_distance=True,
        )


def test_banana_current_penalty_matches_host_current_magnitude_gradient():
    pytest.importorskip(
        "simsoptpp",
        reason="Host CurrentMagnitude parity requires SIMSOPT current objects.",
    )
    from simsopt.field import Current
    from simsopt.field.coil import ScaledCurrent

    from examples.single_stage_optimization.banana_opt.jax_banana_drivers import (
        current_abs_upper_bound_penalty,
    )

    base_current = Current(-1.0)
    scaled_current = ScaledCurrent(base_current, 20.0 * KA_TO_A)
    banana_coils = [_HostCoil(_make_curve(), scaled_current)]
    spec, decision_vector = build_banana_local_objective_spec_from_coils(
        banana_coils,
        stage="stage2",
        weights=Stage2Weights(),
        include_current_penalty=True,
        include_min_length=False,
        include_width=False,
    )
    decision_vector_jax = jnp.asarray(decision_vector, dtype=jnp.float64)
    host_penalty = current_abs_upper_bound_penalty(
        scaled_current,
        spec.banana_current_max_threshold,
    )

    def current_penalty(current_decision_vector):
        return banana_local_terms(spec, current_decision_vector).banana_current_max

    jax_value, jax_grad = jax.value_and_grad(current_penalty)(decision_vector_jax)

    np.testing.assert_allclose(jax_value, host_penalty.J())
    np.testing.assert_allclose(jax_grad[0], host_penalty.dJ()[0])
    np.testing.assert_allclose(
        jax_grad[1:],
        np.zeros(spec.system.decision.curve_dof_count, dtype=np.float64),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("stage", "weights"),
    (
        ("stage2", Stage2Weights()),
        ("single_stage", SingleStageWeights()),
    ),
)
def test_banana_local_terms_match_driver_weighted_terms_for_canonical_biotsavart(
    stage,
    weights,
):
    pytest.importorskip(
        "simsoptpp",
        reason="Driver term parity requires SIMSOPT host coil objects.",
    )
    from examples.single_stage_optimization.banana_opt.jax_banana_drivers import (
        banana_geometry_terms,
        build_biotsavart,
        weighted_sum_objective,
    )
    from simsopt.geo.surfacerzfourier import SurfaceRZFourier
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

    surface = SurfaceRZFourier(
        nfp=1,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.linspace(0.0, 1.0, 8, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 8, endpoint=False),
    )
    surface.set("rc(0,0)", 1.0)
    surface.set("rc(1,0)", 0.2)
    surface.set("zs(1,0)", 0.2)
    biotsavart = build_biotsavart(
        tf_current_ka=-90.0,
        tf_fix_current=True,
        banana_current_ka=20.0,
        banana_fix_current=False,
        banana_order=1,
        banana_dofs=dict(DEFAULT_BANANA_DOFS),
        proxy_current_ka=0.0,
        proxy_rz=DEFAULT_PROXY_RZ,
        vf_current_ka=0.0,
        vf_fix_current=True,
    )
    driver_terms = banana_geometry_terms(
        biotsavart=BiotSavartJAX(biotsavart.coils),
        surface=surface,
        length_weight=weights.length,
        ccdist_weight=weights.ccdist,
        csdist_weight=weights.csdist,
        curvature_weight=weights.curvature,
        poloidal_weight=weights.poloidal,
        width_weight=weights.width,
        self_distance_weight=weights.selfint,
        current_weight=weights.currents,
        include_current_penalties=True,
        include_min_length=True,
        include_width=True,
    )
    spec, decision_vector = build_banana_local_objective_spec_from_biotsavart(
        biotsavart,
        stage=stage,
        weights=weights,
        surface=surface,
        include_current_penalty=True,
        include_min_length=True,
        include_width=True,
        include_coil_surface_distance=True,
    )

    driver_values = {term.name: float(term.objective.J()) for term in driver_terms}
    frozen_terms = banana_local_terms(
        spec,
        jnp.asarray(decision_vector, dtype=jnp.float64),
    )
    expected_terms = {
        "length_max": frozen_terms.length_max,
        "coil_coil_distance": frozen_terms.coil_coil_distance,
        "coil_surface_distance": frozen_terms.coil_surface_distance,
        "curvature": frozen_terms.curvature,
        "poloidal_extent": frozen_terms.poloidal_extent,
        "self_distance": frozen_terms.self_distance,
        "length_min": frozen_terms.length_min,
        "width_max": frozen_terms.width_max,
        "width_min": frozen_terms.width_min,
        "tf_current_max": frozen_terms.tf_current_max,
        "banana_current_max": frozen_terms.banana_current_max,
    }

    assert set(driver_values) == set(expected_terms)
    assert driver_values["length_min"] > 0.0
    assert driver_values["width_max"] > 0.0
    assert driver_values["tf_current_max"] > 0.0
    assert driver_values["banana_current_max"] > 0.0
    for name, expected_value in expected_terms.items():
        np.testing.assert_allclose(
            driver_values[name],
            expected_value,
            rtol=1e-12,
            atol=1e-12,
        )
    expected_weighted_value = sum(
        term.weight * driver_values[term.name]
        for term in driver_terms
        if term.name in driver_values
    )
    np.testing.assert_allclose(
        banana_local_value(spec, jnp.asarray(decision_vector, dtype=jnp.float64)),
        expected_weighted_value,
        rtol=1e-12,
        atol=1e-12,
    )
    driver_objective = weighted_sum_objective(driver_terms)
    frozen_value, frozen_grad = jax.value_and_grad(banana_local_value, argnums=1)(
        spec,
        jnp.asarray(decision_vector, dtype=jnp.float64),
    )

    np.testing.assert_allclose(frozen_value, driver_objective.J(), rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(
        frozen_grad,
        driver_objective.dJ(),
        rtol=1e-12,
        atol=1e-8,
    )


def test_banana_squared_flux_term_matches_squaredflux_jax_for_canonical_biotsavart():
    pytest.importorskip(
        "simsoptpp",
        reason="SquaredFlux parity requires SIMSOPT host coil objects.",
    )
    from examples.single_stage_optimization.banana_opt.jax_banana_drivers import (
        build_biotsavart,
    )
    from simsopt.geo.surfacerzfourier import SurfaceRZFourier
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

    surface = SurfaceRZFourier(
        nfp=1,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.linspace(0.0, 1.0, 8, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 8, endpoint=False),
    )
    surface.set("rc(0,0)", 1.0)
    surface.set("rc(1,0)", 0.2)
    surface.set("zs(1,0)", 0.2)
    surface.fix_all()
    biotsavart = build_biotsavart(
        tf_current_ka=-90.0,
        tf_fix_current=True,
        banana_current_ka=20.0,
        banana_fix_current=False,
        banana_order=1,
        banana_dofs=dict(DEFAULT_BANANA_DOFS),
        proxy_current_ka=0.0,
        proxy_rz=DEFAULT_PROXY_RZ,
        vf_current_ka=0.0,
        vf_fix_current=True,
    )
    squared_flux = SquaredFluxJAX(
        surface,
        BiotSavartJAX(biotsavart.coils),
        definition="normalized",
    )
    weights = replace(
        Stage2Weights(),
        sqflux=3.0,
        length=0.0,
        ccdist=0.0,
        csdist=0.0,
        curvature=0.0,
        poloidal=0.0,
        width=0.0,
        selfint=0.0,
        currents=0.0,
    )
    spec, decision_vector = build_banana_local_objective_spec_from_biotsavart(
        biotsavart,
        stage="stage2",
        weights=weights,
        surface=surface,
        include_current_penalty=False,
        include_min_length=False,
        include_width=False,
        include_coil_surface_distance=False,
        include_squared_flux=True,
    )

    assert spec.include_squared_flux is True
    assert spec.squared_flux_weight == pytest.approx(3.0)
    assert spec.system.fixed_auxiliary_coil_roles == (
        ("tf",) * N_TF + ("proxy",) * N_PROXY + ("vf",) * N_VF
    )
    terms = banana_local_terms(
        spec,
        jnp.asarray(decision_vector, dtype=jnp.float64),
    )
    value, grad = jax.value_and_grad(banana_local_value, argnums=1)(
        spec,
        jnp.asarray(decision_vector, dtype=jnp.float64),
    )

    np.testing.assert_allclose(terms.squared_flux, squared_flux.J(), rtol=1e-11)
    np.testing.assert_allclose(value, 3.0 * squared_flux.J(), rtol=1e-11)
    np.testing.assert_allclose(grad, 3.0 * squared_flux.dJ(), rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize(
    ("stage", "weights"),
    (
        ("stage2", Stage2Weights()),
        ("single_stage", _stage2_weights_with_single_stage_geometry()),
    ),
)
def test_banana_stage2_value_and_grad_matches_driver_full_stage2_terms(
    stage,
    weights,
):
    pytest.importorskip(
        "simsoptpp",
        reason="Full Stage 2 parity requires SIMSOPT host coil objects.",
    )
    from examples.single_stage_optimization.banana_opt.jax_banana_drivers import (
        build_biotsavart,
        build_stage2_objective,
        weighted_sum_objective,
    )
    from simsopt.geo.surfacerzfourier import SurfaceRZFourier
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

    surface = SurfaceRZFourier(
        nfp=1,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.linspace(0.0, 1.0, 8, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 8, endpoint=False),
    )
    surface.set("rc(0,0)", 1.0)
    surface.set("rc(1,0)", 0.2)
    surface.set("zs(1,0)", 0.2)
    surface.fix_all()
    biotsavart = build_biotsavart(
        tf_current_ka=-90.0,
        tf_fix_current=True,
        banana_current_ka=20.0,
        banana_fix_current=False,
        banana_order=1,
        banana_dofs=dict(DEFAULT_BANANA_DOFS),
        proxy_current_ka=0.0,
        proxy_rz=DEFAULT_PROXY_RZ,
        vf_current_ka=0.0,
        vf_fix_current=True,
    )
    biotsavart_jax = BiotSavartJAX(biotsavart.coils)
    driver_objective, driver_terms = build_stage2_objective(
        biotsavart_jax=biotsavart_jax,
        surface=surface,
        weights=weights,
        include_current_penalties=True,
        include_min_length=True,
        include_width=True,
    )
    spec, decision_vector = build_banana_local_objective_spec_from_biotsavart(
        biotsavart,
        stage=stage,
        weights=weights,
        surface=surface,
        include_current_penalty=True,
        include_min_length=True,
        include_width=True,
        include_coil_surface_distance=True,
        include_squared_flux=True,
    )

    frozen_terms = banana_local_terms(
        spec,
        jnp.asarray(decision_vector, dtype=jnp.float64),
    )
    expected_terms = {
        "squared_flux": frozen_terms.squared_flux,
        "length_max": frozen_terms.length_max,
        "coil_coil_distance": frozen_terms.coil_coil_distance,
        "coil_surface_distance": frozen_terms.coil_surface_distance,
        "curvature": frozen_terms.curvature,
        "poloidal_extent": frozen_terms.poloidal_extent,
        "self_distance": frozen_terms.self_distance,
        "length_min": frozen_terms.length_min,
        "width_max": frozen_terms.width_max,
        "width_min": frozen_terms.width_min,
        "tf_current_max": frozen_terms.tf_current_max,
        "banana_current_max": frozen_terms.banana_current_max,
    }
    driver_values = {term.name: float(term.objective.J()) for term in driver_terms}

    assert set(driver_values) == set(expected_terms)
    np.testing.assert_allclose(driver_objective.x, decision_vector)
    for name, expected_value in expected_terms.items():
        np.testing.assert_allclose(
            driver_values[name],
            expected_value,
            rtol=1e-10,
            atol=1e-8,
        )
    alignment_weight = 0.25

    def driver_full_vector_term_grad(term_name):
        objective_with_term = weighted_sum_objective(
            [
                replace(
                    candidate,
                    weight=1.0 if candidate.name == term_name else alignment_weight,
                )
                for candidate in driver_terms
            ]
        )
        objective_without_term = weighted_sum_objective(
            [
                replace(candidate, weight=alignment_weight)
                for candidate in driver_terms
            ]
        )
        np.testing.assert_allclose(objective_with_term.x, decision_vector)
        np.testing.assert_allclose(objective_without_term.x, decision_vector)
        return (
            objective_with_term.dJ() - objective_without_term.dJ()
        ) / (1.0 - alignment_weight)

    for term in driver_terms:
        term_value_and_grad = jax.value_and_grad(
            lambda dofs, term_name=term.name: getattr(
                banana_local_terms(spec, dofs),
                term_name,
            ),
        )
        term_value, term_grad = term_value_and_grad(
            jnp.asarray(decision_vector, dtype=jnp.float64)
        )
        np.testing.assert_allclose(
            term_value,
            term.objective.J(),
            rtol=1e-10,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            term_grad,
            driver_full_vector_term_grad(term.name),
            rtol=1e-9,
            atol=1e-7,
        )

    frozen_value, frozen_grad = jax.value_and_grad(banana_local_value, argnums=1)(
        spec,
        jnp.asarray(decision_vector, dtype=jnp.float64),
    )
    np.testing.assert_allclose(frozen_value, driver_objective.J(), rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(frozen_grad, driver_objective.dJ(), rtol=1e-9, atol=1e-7)
    fd_spec = replace(
        spec,
        include_current_penalty=False,
        include_tf_current_penalty=False,
        tf_current_weight=0.0,
        banana_current_weight=0.0,
    )
    _fd_value, fd_grad = jax.value_and_grad(banana_local_value, argnums=1)(
        fd_spec,
        jnp.asarray(decision_vector, dtype=jnp.float64),
    )
    fd_grad_np = np.asarray(fd_grad, dtype=np.float64)
    for dof_index in (0, decision_vector.size - 1):
        step = np.zeros_like(decision_vector)
        step[dof_index] = 1.0e-6
        value_plus = banana_local_value(
            fd_spec,
            jnp.asarray(decision_vector + step, dtype=jnp.float64),
        )
        value_minus = banana_local_value(
            fd_spec,
            jnp.asarray(decision_vector - step, dtype=jnp.float64),
        )
        finite_difference = float((value_plus - value_minus) / (2.0e-6))
        np.testing.assert_allclose(
            fd_grad_np[dof_index],
            finite_difference,
            rtol=1e-4,
            atol=1e-4,
        )


def test_free_tf_and_banana_current_penalty_gradients_match_driver_order():
    pytest.importorskip(
        "simsoptpp",
        reason="Driver current parity requires SIMSOPT host current objects.",
    )
    from examples.single_stage_optimization.banana_opt.jax_banana_drivers import (
        banana_geometry_terms,
        build_biotsavart,
        weighted_sum_objective,
    )
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

    current_only_weights = replace(
        Stage2Weights(),
        length=0.0,
        ccdist=0.0,
        csdist=0.0,
        curvature=0.0,
        poloidal=0.0,
        width=0.0,
        selfint=0.0,
        currents=1.0,
    )
    biotsavart = build_biotsavart(
        tf_current_ka=-90.0,
        tf_fix_current=False,
        banana_current_ka=20.0,
        banana_fix_current=False,
        banana_order=1,
        banana_dofs=dict(DEFAULT_BANANA_DOFS),
        proxy_current_ka=0.0,
        proxy_rz=DEFAULT_PROXY_RZ,
        vf_current_ka=0.0,
        vf_fix_current=True,
    )
    driver_terms = banana_geometry_terms(
        biotsavart=BiotSavartJAX(biotsavart.coils),
        surface=None,
        length_weight=current_only_weights.length,
        ccdist_weight=current_only_weights.ccdist,
        csdist_weight=current_only_weights.csdist,
        curvature_weight=current_only_weights.curvature,
        poloidal_weight=current_only_weights.poloidal,
        width_weight=current_only_weights.width,
        self_distance_weight=current_only_weights.selfint,
        current_weight=current_only_weights.currents,
        include_current_penalties=True,
        include_min_length=False,
        include_width=False,
    )
    driver_current_terms = [
        term
        for term in driver_terms
        if term.name in ("tf_current_max", "banana_current_max")
    ]
    driver_objective = weighted_sum_objective(driver_current_terms)
    spec, decision_vector = build_banana_local_objective_spec_from_biotsavart(
        biotsavart,
        stage="stage2",
        weights=current_only_weights,
        include_current_penalty=True,
        include_min_length=False,
        include_width=False,
        include_coil_surface_distance=False,
    )

    assert spec.system.decision.tf_current_dof_count == 1
    assert spec.system.decision.current_dof_count == 1
    np.testing.assert_allclose(driver_objective.x, decision_vector[:2])

    def frozen_current_value(frozen_decision_vector):
        terms = banana_local_terms(spec, frozen_decision_vector)
        return (
            spec.tf_current_weight * terms.tf_current_max
            + spec.banana_current_weight * terms.banana_current_max
        )

    frozen_value, frozen_grad = jax.value_and_grad(frozen_current_value)(
        jnp.asarray(decision_vector, dtype=jnp.float64)
    )

    np.testing.assert_allclose(frozen_value, driver_objective.J())
    np.testing.assert_allclose(
        frozen_grad[:2],
        driver_objective.dJ(),
        rtol=1e-12,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        frozen_grad[2:],
        np.zeros(spec.system.decision.curve_dof_count, dtype=np.float64),
        atol=1e-12,
    )


def test_banana_local_spec_decision_vector_matches_stage2_target_split():
    banana_coils = _make_banana_coils()
    spec, decision_vector = build_banana_local_objective_spec_from_coils(
        banana_coils,
        stage="stage2",
        weights=Stage2Weights(),
        include_current_penalty=True,
        include_min_length=True,
        include_width=True,
    )

    state = stage2_target_optimizer_state_from_dofs(
        decision_vector,
        curve_dof_count=spec.system.decision.curve_dof_count,
    )

    np.testing.assert_allclose(state.current_dof, np.array(decision_vector[0]))
    np.testing.assert_allclose(state.curve_dofs, decision_vector[1:])


def test_build_banana_local_spec_from_coils_freezes_fixed_current_out_of_decision():
    curve = _make_curve()
    fixed_current = _HostCurrent(full_value=1.5, free_value=None)
    spec, decision_vector = build_banana_local_objective_spec_from_coils(
        _make_banana_coils(curve=curve, current=fixed_current),
        stage="stage2",
        weights=Stage2Weights(),
        include_current_penalty=False,
        include_min_length=False,
        include_width=False,
    )

    assert spec.system.decision.current_dof_count == 0
    assert spec.system.decision.tf_current_dof_count == 0
    np.testing.assert_allclose(spec.system.fixed_current_dofs, np.array([1.5]))
    np.testing.assert_allclose(decision_vector, curve.get_dofs())
    geometry = banana_geometry_from_dofs(
        spec.system,
        jnp.asarray(decision_vector, dtype=jnp.float64),
    )
    np.testing.assert_allclose(geometry.currents, np.array([24000.0, -24000.0]))


def test_build_banana_local_spec_from_coils_rejects_non_shared_base_curve():
    banana_coils = _make_banana_coils()
    banana_coils[1] = _HostCoil(
        _RotatedCurve(_make_curve(), np.eye(3, dtype=np.float64)),
        banana_coils[1].current,
    )

    with pytest.raises(ValueError, match="share one base curve"):
        build_banana_local_objective_spec_from_coils(
            banana_coils,
            stage="stage2",
            weights=Stage2Weights(),
            include_current_penalty=True,
            include_min_length=True,
            include_width=True,
        )


def test_build_banana_local_spec_from_coils_rejects_multi_current_layout():
    with pytest.raises(ValueError, match="0 or 1 free base current"):
        build_banana_local_objective_spec_from_coils(
            _make_banana_coils(current=_BadVectorCurrent()),
            stage="stage2",
            weights=Stage2Weights(),
            include_current_penalty=True,
            include_min_length=True,
            include_width=True,
        )


def test_build_banana_local_spec_from_biotsavart_uses_canonical_block():
    banana_coils = _make_banana_coils()
    tf_current = _HostCurrent(full_value=2.0, free_value=2.0)
    filler = _HostCoil(_make_curve(), _ScaledCurrent(tf_current, -45.0 * KA_TO_A))
    biotsavart = _HostBiotSavart(
        coils=[filler for _ in range(BANANA_IDX)]
        + [banana_coils[index % len(banana_coils)] for index in range(N_BANANA)]
    )

    spec, decision_vector = build_banana_local_objective_spec_from_biotsavart(
        biotsavart,
        stage="stage2",
        weights=Stage2Weights(),
        include_current_penalty=True,
        include_min_length=True,
        include_width=True,
    )

    assert spec.system.decision.tf_current_dof_count == 1
    assert spec.system.decision.current_dof_count == 1
    assert len(spec.system.symmetries) == N_BANANA
    np.testing.assert_allclose(decision_vector[0], np.array(2.0))
    np.testing.assert_allclose(decision_vector[1], np.array(1.25))
    geometry = banana_geometry_from_dofs(
        spec.system,
        jnp.asarray(decision_vector, dtype=jnp.float64),
    )
    np.testing.assert_allclose(geometry.tf_current, np.array(-90.0 * KA_TO_A))


def test_build_banana_local_spec_from_biotsavart_rejects_incomplete_block():
    biotsavart = _HostBiotSavart(coils=[])

    with pytest.raises(ValueError, match="block is incomplete"):
        build_banana_local_objective_spec_from_biotsavart(
            biotsavart,
            stage="stage2",
            weights=Stage2Weights(),
            include_current_penalty=True,
            include_min_length=True,
            include_width=True,
        )
