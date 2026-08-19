"""Typed objective and Boozer-construction policy for the flat-675 problem.

The policy is the single owner of the mapping from the frozen campaign's
physical weights, targets, and thresholds onto the shared single-stage
outer-objective configuration keys, so a weight only ever has to be renamed
here.  Values are host scalars; nothing in this module traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np

from .formulation import Flat675ContractError, flat675_finite_float

# Canonical assembly order of the eight production terms, expressed in the
# shared-math term keys.  The certified objective is the ordered sum over this
# tuple, so its order is part of the numerical contract and not cosmetic.
FLAT675_OBJECTIVE_TERM_KEYS: Final[tuple[str, ...]] = (
    "non_qs",
    "residual",
    "iota",
    "length",
    "curve_curve",
    "curve_surface",
    "surface_vessel",
    "curvature",
)


class Flat675BoozerLabelType(str, Enum):
    """Surface label the Boozer residual term constrains."""

    VOLUME = "volume"
    AREA = "area"
    TOROIDAL_FLUX = "toroidal_flux"


def _nonnegative_float(value: object, where: str) -> float:
    scalar = flat675_finite_float(value, where)
    if scalar < 0.0:
        raise Flat675ContractError(f"{where} must be nonnegative.")
    return scalar


def _positive_float(value: object, where: str) -> float:
    scalar = flat675_finite_float(value, where)
    if scalar <= 0.0:
        raise Flat675ContractError(f"{where} must be positive.")
    return scalar


def _nonnegative_index(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Flat675ContractError(f"{where} must be a nonnegative integer.")
    return int(value)


def _positive_integer(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Flat675ContractError(f"{where} must be a positive integer.")
    return int(value)


@dataclass(frozen=True, slots=True)
class Flat675HardwareSoftPenaltyPolicy:
    """Shared scale, bases, thresholds, and exponent of the hardware penalties.

    The coil-coil-distance and curvature terms are the two hardware penalties
    whose weights are a common scale times a base weight; this record owns
    that factorization so the objective policy cannot drift from it.
    """

    common_scale: float
    curve_curve_base_weight: float
    curvature_base_weight: float
    curve_curve_threshold_m: float
    curvature_threshold_inverse_m: float
    penalty_exponent: float

    def __post_init__(self) -> None:
        for field_name in (
            "common_scale",
            "curve_curve_base_weight",
            "curvature_base_weight",
            "curve_curve_threshold_m",
            "curvature_threshold_inverse_m",
            "penalty_exponent",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_float(
                    getattr(self, field_name),
                    f"hardware_soft_penalty_policy.{field_name}",
                ),
            )

    @property
    def curve_curve_effective_weight(self) -> float:
        return self.common_scale * self.curve_curve_base_weight

    @property
    def curvature_effective_weight(self) -> float:
        return self.common_scale * self.curvature_base_weight


PRODUCTION_FLAT675_HARDWARE_SOFT_PENALTY_POLICY: Final[
    Flat675HardwareSoftPenaltyPolicy
] = Flat675HardwareSoftPenaltyPolicy(
    common_scale=10.0,
    curve_curve_base_weight=100.0,
    curvature_base_weight=0.1,
    curve_curve_threshold_m=0.05,
    curvature_threshold_inverse_m=40.0,
    penalty_exponent=2.0,
)


@dataclass(frozen=True, slots=True)
class Flat675ObjectivePolicy:
    """Weights, targets, and thresholds of the eight production terms.

    Construction fails closed when the hardware-penalty material disagrees
    with ``hardware_soft_penalty_policy``: the coil-coil and curvature weights
    and thresholds have two sources in the frozen artifacts, and a run whose
    two sources disagree is not the certified objective.
    """

    iota_target: float
    boozer_constraint_weight: float
    boozer_target_label: float
    boozer_label_type: Flat675BoozerLabelType
    boozer_label_phi_index: int
    non_qs_weight: float
    non_qs_grid_size: int
    residual_weight: float
    iota_weight: float
    length_weight: float
    length_target_m: float
    curve_curve_weight: float
    curve_curve_threshold_m: float
    curve_surface_weight: float
    curve_surface_threshold_m: float
    surface_vessel_weight: float
    surface_vessel_threshold_m: float
    curvature_weight: float
    curvature_threshold_inverse_m: float
    optimized_coil_index: int
    hardware_soft_penalty_policy: Flat675HardwareSoftPenaltyPolicy

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "iota_target",
            flat675_finite_float(self.iota_target, "objective_policy.iota_target"),
        )
        if not isinstance(self.boozer_label_type, Flat675BoozerLabelType):
            raise Flat675ContractError(
                "objective_policy.boozer_label_type must be Flat675BoozerLabelType."
            )
        if not isinstance(
            self.hardware_soft_penalty_policy, Flat675HardwareSoftPenaltyPolicy
        ):
            raise Flat675ContractError(
                "objective_policy.hardware_soft_penalty_policy has the wrong type."
            )
        for field_name in (
            "boozer_label_phi_index",
            "optimized_coil_index",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_index(
                    getattr(self, field_name), f"objective_policy.{field_name}"
                ),
            )
        object.__setattr__(
            self,
            "non_qs_grid_size",
            _positive_integer(
                self.non_qs_grid_size, "objective_policy.non_qs_grid_size"
            ),
        )
        for field_name in (
            "boozer_constraint_weight",
            "non_qs_weight",
            "residual_weight",
            "iota_weight",
            "length_weight",
            "curve_curve_weight",
            "curve_surface_weight",
            "surface_vessel_weight",
            "curvature_weight",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_float(
                    getattr(self, field_name), f"objective_policy.{field_name}"
                ),
            )
        for field_name in (
            "boozer_target_label",
            "length_target_m",
            "curve_curve_threshold_m",
            "curve_surface_threshold_m",
            "surface_vessel_threshold_m",
            "curvature_threshold_inverse_m",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_float(
                    getattr(self, field_name), f"objective_policy.{field_name}"
                ),
            )
        self._require_hardware_penalty_agreement()

    def _require_hardware_penalty_agreement(self) -> None:
        soft_penalty = self.hardware_soft_penalty_policy
        observed = (
            self.curve_curve_weight,
            self.curvature_weight,
            self.curve_curve_threshold_m,
            self.curvature_threshold_inverse_m,
        )
        expected = (
            soft_penalty.curve_curve_effective_weight,
            soft_penalty.curvature_effective_weight,
            soft_penalty.curve_curve_threshold_m,
            soft_penalty.curvature_threshold_inverse_m,
        )
        if observed != expected:
            raise Flat675ContractError(
                "objective_policy hardware penalties differ from the frozen "
                f"hardware soft-penalty policy: {observed!r} != {expected!r}."
            )

    @property
    def curvature_p_norm(self) -> float:
        """Exponent of the curvature penalty, owned by the soft-penalty policy."""
        return self.hardware_soft_penalty_policy.penalty_exponent


@dataclass(frozen=True, slots=True)
class Flat675BoozerSystemPolicy:
    """Construction policy for the two-column Boozer least-squares system."""

    weight_by_inverse_field_magnitude: bool

    def __post_init__(self) -> None:
        if type(self.weight_by_inverse_field_magnitude) is not bool:
            raise Flat675ContractError(
                "weight_by_inverse_field_magnitude must be exactly bool."
            )


def flat675_outer_objective_config(
    policy: Flat675ObjectivePolicy,
    *,
    nfp: int,
    vessel_gamma: object,
) -> dict[str, object]:
    """Render the policy as the shared outer-objective configuration mapping.

    ``vessel_gamma`` enters the mapping as it is given: passing a traced array
    keeps the three vessel DOFs differentiably connected to the
    surface-to-vessel term, which is how the flat formulation optimizes them.
    """
    grid_size = policy.non_qs_grid_size
    return {
        "non_qs_weight": policy.non_qs_weight,
        "non_qs_quadpoints_phi": np.linspace(
            0.0,
            1.0 / int(nfp),
            grid_size,
            endpoint=False,
            dtype=np.float64,
        ),
        "non_qs_quadpoints_theta": np.linspace(
            0.0,
            1.0,
            grid_size,
            endpoint=False,
            dtype=np.float64,
        ),
        "non_qs_axis": 0,
        "residual_weight": policy.residual_weight,
        "iota_weight": policy.iota_weight,
        "length_weight": policy.length_weight,
        "length_target": policy.length_target_m,
        "curve_curve_weight": policy.curve_curve_weight,
        "curve_curve_threshold": policy.curve_curve_threshold_m,
        "curve_surface_weight": policy.curve_surface_weight,
        "curve_surface_threshold": policy.curve_surface_threshold_m,
        "surface_vessel_weight": policy.surface_vessel_weight,
        "surface_vessel_threshold": policy.surface_vessel_threshold_m,
        "curvature_weight": policy.curvature_weight,
        "curvature_threshold": policy.curvature_threshold_inverse_m,
        "curvature_p_norm": policy.curvature_p_norm,
        "optimized_coil_index": policy.optimized_coil_index,
        "vessel_gamma": vessel_gamma,
    }


__all__ = [
    "FLAT675_OBJECTIVE_TERM_KEYS",
    "PRODUCTION_FLAT675_HARDWARE_SOFT_PENALTY_POLICY",
    "Flat675BoozerLabelType",
    "Flat675BoozerSystemPolicy",
    "Flat675HardwareSoftPenaltyPolicy",
    "Flat675ObjectivePolicy",
    "flat675_outer_objective_config",
]
