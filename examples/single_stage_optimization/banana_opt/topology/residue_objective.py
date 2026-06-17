from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np

from simsopt._core.derivative import Derivative, derivative_dec
from simsopt._core.optimizable import Optimizable
from simsopt.field.biotsavart import BiotSavart

from .fieldline_map import (
    DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    FieldlineIntegratorOptions,
)
from .periodic_orbit import (
    BRANCH_STATUS_BAD_DETERMINANT,
    BRANCH_STATUS_BRANCH_MISMATCH,
    BRANCH_STATUS_CONVERGED,
    BRANCH_STATUS_INTEGRATION_FAILED,
    BRANCH_STATUS_MAX_ITERATIONS,
    BRANCH_STATUS_NEWTON_STALLED,
    BRANCH_STATUS_OUTSIDE_RADIAL_WINDOW,
    BRANCH_STATUS_WRONG_WINDING,
    DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
    PeriodicOrbitResult,
    PeriodicOrbitSolverOptions,
)
from .poincare_chart import PoincareChart
from .rational_target import (
    GREENE_BRANCHES,
    GREENE_IOTA_CONVENTION,
    GREENE_MAP_CONVENTION_FULL_TORUS,
    RationalTarget,
)
from .residue_sensitivity import (
    BIOT_SAVART_BRANCH_RESIDUE_DOF_GRADIENT_METHOD,
    BIOT_SAVART_BRANCH_RESIDUE_LOCAL_SENSITIVITY_LIMITATIONS,
    BIOT_SAVART_BRANCH_RESIDUE_LOCAL_SENSITIVITY_METHOD,
    BIOT_SAVART_BRANCH_RESOLVED_VJP_TAYLOR_MODE,
    DEFAULT_RESIDUE_SATISFIED_THRESHOLD,
    BiotSavartBranchResidueVjpDiagnostic,
    branch_resolved_biot_savart_residue_vjp,
    solve_biot_savart_residue_branch,
)


GREENE_RESIDUE_OBJECTIVE_SCHEMA_VERSION = "greene_residue_objective_v1"
GREENE_RESIDUE_OBJECTIVE_VALIDATION_STATUS_PASSED = "passed"
GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_MIN_ORDER = 1.95
GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_MIN_SAMPLES = 3
GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_ORDER_RTOL = 1.0e-9
GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_ORDER_ATOL = 1.0e-12
GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_RESIDUAL_RTOL = 1.0e-9
GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_RESIDUAL_ATOL = 1.0e-18
GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_WINDING_ATOL = 1.0e-4
DEFAULT_RESIDUE_OBJECTIVE_WEIGHT = 0.0
DEFAULT_RESIDUE_OBJECTIVE_SAMPLES_PER_FULL_TORUS = 768
GREENE_RESIDUE_OBJECTIVE_VALIDATION_EVIDENCE_KIND = "seed_json_payload_consistency_gate"
GREENE_RESIDUE_OBJECTIVE_VALIDATION_LIMITATIONS = (
    "loader_validates_json_proof_payloads_not_fresh_field_solves",
    "no_bundled_landreman_2106_14930_poincare_or_spec_reproduction",
)
RESIDUE_TARGET_PAYLOAD_KEYS = frozenset(
    {
        "p",
        "q",
        "weight",
        "radial_label",
        "radial_window",
        "branches",
        "phi0",
        "nfp",
        "convention",
        "fourier_m",
        "fourier_n",
        "map_convention",
    }
)

GREENE_RESIDUE_BRANCH_LOSS_RAISE = "raise"
GREENE_RESIDUE_BRANCH_LOSS_TREAT_AS_SATISFIED = "treat_as_satisfied"
GREENE_RESIDUE_BRANCH_LOSS_POLICIES = frozenset(
    {
        GREENE_RESIDUE_BRANCH_LOSS_RAISE,
        GREENE_RESIDUE_BRANCH_LOSS_TREAT_AS_SATISFIED,
    }
)
# Non-converged solver statuses that mean the targeted island can no longer be
# located as a valid fixed point of the requested type/winding/radial window --
# the geometric "island healed or drifted away" outcome. Under the
# treat_as_satisfied policy these contribute zero to J and dJ (the removal goal
# is met, or the objective can no longer push on this resonance). The numerical
# -integrity statuses BRANCH_STATUS_BAD_DETERMINANT and
# BRANCH_STATUS_INTEGRATION_FAILED are deliberately excluded so a broken
# computation (under-resolved symplectic map, blown-up integration) stays loud
# regardless of policy, as do any exceptions raised inside the branch solve.
GREENE_RESIDUE_BRANCH_LOSS_STATUSES = frozenset(
    {
        BRANCH_STATUS_MAX_ITERATIONS,
        BRANCH_STATUS_NEWTON_STALLED,
        BRANCH_STATUS_OUTSIDE_RADIAL_WINDOW,
        BRANCH_STATUS_WRONG_WINDING,
        BRANCH_STATUS_BRANCH_MISMATCH,
    }
)
# Non-converged statuses that signal a broken computation rather than a healed
# island, and therefore stay loud regardless of the on_branch_loss policy.
GREENE_RESIDUE_BRANCH_INTEGRITY_FAILURE_STATUSES = frozenset(
    {
        BRANCH_STATUS_BAD_DETERMINANT,
        BRANCH_STATUS_INTEGRATION_FAILED,
    }
)
DEFAULT_RESIDUE_OBJECTIVE_ON_BRANCH_LOSS = GREENE_RESIDUE_BRANCH_LOSS_RAISE

ResidueBranchKey = tuple[str, str]
SectionState = tuple[float, float]


@dataclass(frozen=True, slots=True)
class ResidueBranchSeed:
    """Validated fixed-point seed for one target manifest and O/X branch."""

    target_id: str
    branch: str
    section_state: Sequence[float]
    validation_id: str
    optimizer_taylor_validated: bool = False
    direct_proxy_consistency_validated: bool = False
    real_field_nonzero_winding_validated: bool = False

    def __post_init__(self) -> None:
        target_id = str(self.target_id)
        branch = str(self.branch)
        validation_id = str(self.validation_id)
        optimizer_taylor_validated = _seed_bool_value(
            "optimizer_taylor_validated",
            self.optimizer_taylor_validated,
        )
        direct_proxy_consistency_validated = _seed_bool_value(
            "direct_proxy_consistency_validated",
            self.direct_proxy_consistency_validated,
        )
        real_field_nonzero_winding_validated = _seed_bool_value(
            "real_field_nonzero_winding_validated",
            self.real_field_nonzero_winding_validated,
        )
        if target_id == "":
            raise ValueError("Greene residue branch seed target_id must be nonempty")
        if branch not in GREENE_BRANCHES:
            raise ValueError(f"Unknown Greene residue branch label: {branch}")
        if validation_id == "":
            raise ValueError(
                "Greene residue branch seed validation_id must be nonempty"
            )
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "branch", branch)
        object.__setattr__(
            self,
            "section_state",
            _normalize_section_state(self.section_state),
        )
        object.__setattr__(self, "validation_id", validation_id)
        object.__setattr__(
            self,
            "optimizer_taylor_validated",
            optimizer_taylor_validated,
        )
        object.__setattr__(
            self,
            "direct_proxy_consistency_validated",
            direct_proxy_consistency_validated,
        )
        object.__setattr__(
            self,
            "real_field_nonzero_winding_validated",
            real_field_nonzero_winding_validated,
        )

    @property
    def key(self) -> ResidueBranchKey:
        return (self.target_id, self.branch)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "branch": self.branch,
            "section_state": list(self.section_state),
            "validation_id": self.validation_id,
            "optimizer_taylor_validated": self.optimizer_taylor_validated,
            "direct_proxy_consistency_validated": (
                self.direct_proxy_consistency_validated
            ),
            "real_field_nonzero_winding_validated": (
                self.real_field_nonzero_winding_validated
            ),
        }


@dataclass(frozen=True, slots=True)
class ResidueObjectiveBranchValue:
    target_id: str
    branch: str
    residue: float
    status: str
    monodromy: tuple[tuple[float, float], tuple[float, float]]
    trace_m: float
    det_m: float
    winding: float
    radial_label: float
    closure_residual_norm: float
    min_bphi_over_b: float
    residue_scale: float
    target_weight: float
    objective_weight: float
    branch_objective: float
    state: SectionState

    def to_json_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "branch": self.branch,
            "residue": self.residue,
            "status": self.status,
            "monodromy": [list(row) for row in self.monodromy],
            "trace_m": self.trace_m,
            "det_m": self.det_m,
            "winding": self.winding,
            "radial_label": self.radial_label,
            "closure_residual_norm": self.closure_residual_norm,
            "min_Bphi_over_B": self.min_bphi_over_b,
            "residue_scale": self.residue_scale,
            "target_weight": self.target_weight,
            "objective_weight": self.objective_weight,
            "branch_objective": self.branch_objective,
            "state": list(self.state),
        }


class BiotSavartGreeneResidueObjective(Optimizable):
    """Opt-in direct-BiotSavart Greene residue objective.

    The objective is zero by default. A nonzero weight requires explicit
    target-manifest seeds from a prior validated branch solve.
    """

    def __init__(
        self,
        biot_savart: BiotSavart,
        *,
        targets: Sequence[RationalTarget],
        chart: PoincareChart,
        branch_seeds: Sequence[ResidueBranchSeed] = (),
        validation_id: str = "",
        objective_weight: float = DEFAULT_RESIDUE_OBJECTIVE_WEIGHT,
        residue_scale: float = 1.0,
        target_manifest_id: str | None = None,
        integrator_options: FieldlineIntegratorOptions = (
            DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS
        ),
        solver_options: PeriodicOrbitSolverOptions = (
            DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS
        ),
        r_satisfied: float = DEFAULT_RESIDUE_SATISFIED_THRESHOLD,
        local_difference_step: float = 1.0e-6,
        on_branch_loss: str = DEFAULT_RESIDUE_OBJECTIVE_ON_BRANCH_LOSS,
    ) -> None:
        if not isinstance(biot_savart, BiotSavart):
            raise TypeError("Greene residue objective requires direct BiotSavart")
        on_branch_loss_policy = str(on_branch_loss)
        if on_branch_loss_policy not in GREENE_RESIDUE_BRANCH_LOSS_POLICIES:
            raise ValueError(
                "Greene residue objective on_branch_loss must be one of "
                f"{sorted(GREENE_RESIDUE_BRANCH_LOSS_POLICIES)}, got "
                f"{on_branch_loss_policy!r}"
            )
        targets_tuple = tuple(targets)
        if len(targets_tuple) == 0:
            raise ValueError("Greene residue objective requires at least one target")
        objective_weight_value = float(objective_weight)
        if objective_weight_value < 0.0 or not isfinite(objective_weight_value):
            raise ValueError(
                "Greene residue objective weight must be finite and nonnegative"
            )
        residue_scale_value = float(residue_scale)
        if residue_scale_value <= 0.0 or not isfinite(residue_scale_value):
            raise ValueError("Greene residue objective residue_scale must be positive")
        validation_label = str(validation_id)
        branch_seeds_tuple = tuple(branch_seeds)
        if objective_weight_value > 0.0 and validation_label == "":
            raise ValueError("Greene residue objective validation_id must be nonempty")
        if (
            objective_weight_value == 0.0
            and branch_seeds_tuple
            and validation_label == ""
        ):
            raise ValueError(
                "Greene residue objective validation_id must be nonempty when "
                "branch seeds are provided"
            )
        computed_target_manifest_id = residue_objective_target_manifest_id(
            targets_tuple
        )
        if (
            target_manifest_id is not None
            and str(target_manifest_id) != computed_target_manifest_id
        ):
            raise ValueError(
                "Greene residue objective target_manifest_id does not match targets"
            )

        Optimizable.__init__(self, depends_on=[biot_savart])
        self.biot_savart = biot_savart
        self.targets = targets_tuple
        self.chart = chart
        self.validation_id = validation_label
        self.target_manifest_id = computed_target_manifest_id
        self.objective_weight = objective_weight_value
        self.residue_scale = residue_scale_value
        self.integrator_options = integrator_options
        self.solver_options = solver_options
        self.r_satisfied = float(r_satisfied)
        self.local_difference_step = float(local_difference_step)
        self.on_branch_loss = on_branch_loss_policy
        requires_branch_validation = objective_weight_value > 0.0 or bool(
            branch_seeds_tuple
        )
        self._branch_state_by_key = (
            _validated_seed_state_map(
                targets_tuple,
                branch_seeds_tuple,
                validation_id=validation_label,
                require_optimizer_taylor=objective_weight_value > 0.0,
            )
            if requires_branch_validation
            else {}
        )
        self._J: float | None = None
        self._dJ: Derivative | None = None
        self._branch_values: tuple[ResidueObjectiveBranchValue, ...] | None = None
        self._lost_branch_keys: frozenset[ResidueBranchKey] | None = None
        self._gradient_diagnostics: (
            tuple[BiotSavartBranchResidueVjpDiagnostic, ...] | None
        ) = None

    def J(self) -> float:
        if self._J is None:
            self._compute_value()
        return float(self._J)

    @derivative_dec
    def dJ(self) -> Derivative:
        if self._dJ is None:
            self._compute_gradient()
        return self._dJ

    def recompute_bell(self, parent=None) -> None:
        self._J = None
        self._dJ = None
        self._branch_values = None
        self._lost_branch_keys = None
        self._gradient_diagnostics = None

    def branch_values(self) -> tuple[ResidueObjectiveBranchValue, ...]:
        if self._branch_values is None:
            self._compute_value()
        return self._branch_values

    def gradient_diagnostics(self) -> tuple[BiotSavartBranchResidueVjpDiagnostic, ...]:
        if self._gradient_diagnostics is None:
            self._compute_gradient()
        return self._gradient_diagnostics

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": GREENE_RESIDUE_OBJECTIVE_SCHEMA_VERSION,
            "enabled": self.objective_weight > 0.0,
            "target_manifest_id": self.target_manifest_id,
            "validation_id": self.validation_id,
            "validation_evidence_kind": (
                GREENE_RESIDUE_OBJECTIVE_VALIDATION_EVIDENCE_KIND
            ),
            "validation_limitations": list(
                GREENE_RESIDUE_OBJECTIVE_VALIDATION_LIMITATIONS
            ),
            "dof_gradient_method": BIOT_SAVART_BRANCH_RESIDUE_DOF_GRADIENT_METHOD,
            "local_sensitivity_method": (
                BIOT_SAVART_BRANCH_RESIDUE_LOCAL_SENSITIVITY_METHOD
            ),
            "local_sensitivity_limitations": list(
                BIOT_SAVART_BRANCH_RESIDUE_LOCAL_SENSITIVITY_LIMITATIONS
            ),
            "objective_weight": self.objective_weight,
            "residue_scale": self.residue_scale,
            "on_branch_loss": self.on_branch_loss,
            "value": self.J(),
            "branches": [value.to_json_dict() for value in self.branch_values()],
        }

    def _compute_value(self) -> None:
        if self.objective_weight == 0.0:
            self._J = 0.0
            self._branch_values = ()
            self._lost_branch_keys = frozenset()
            return

        branch_values: list[ResidueObjectiveBranchValue] = []
        lost_branch_keys: list[ResidueBranchKey] = []
        objective_value = 0.0
        for target in self.targets:
            target_id = target.manifest_key()
            for branch in target.branches:
                key = (target_id, branch)
                result = solve_biot_savart_residue_branch(
                    self.biot_savart,
                    self._branch_state_by_key[key],
                    target=target,
                    chart=self.chart,
                    branch=branch,
                    integrator_options=self.integrator_options,
                    solver_options=self.solver_options,
                )
                if result.status != BRANCH_STATUS_CONVERGED:
                    if not self._branch_loss_is_satisfied(result.status):
                        raise ValueError(self._branch_loss_message(target, key, result))
                    lost_branch_keys.append(key)
                    branch_values.append(
                        self._branch_value(target, target_id, branch, result, 0.0)
                    )
                    continue
                residue = float(result.residue_diagnostic.residue)
                branch_objective = self._branch_objective(target, residue)
                objective_value += branch_objective
                branch_values.append(
                    self._branch_value(
                        target, target_id, branch, result, branch_objective
                    )
                )
        self._J = float(objective_value)
        self._branch_values = tuple(branch_values)
        self._lost_branch_keys = frozenset(lost_branch_keys)

    def _branch_loss_is_satisfied(self, status: str) -> bool:
        """True when a non-converged branch should count as satisfied (J += 0).

        Only the geometric loss statuses qualify, and only under the
        treat_as_satisfied policy; numerical-integrity failures always fall
        through to the loud raise.
        """
        return (
            self.on_branch_loss == GREENE_RESIDUE_BRANCH_LOSS_TREAT_AS_SATISFIED
            and status in GREENE_RESIDUE_BRANCH_LOSS_STATUSES
        )

    def _branch_value(
        self,
        target: RationalTarget,
        target_id: str,
        branch: str,
        result: PeriodicOrbitResult,
        branch_objective: float,
    ) -> ResidueObjectiveBranchValue:
        monodromy = np.asarray(result.tangent_result.monodromy, dtype=float)
        return ResidueObjectiveBranchValue(
            target_id=target_id,
            branch=branch,
            residue=float(result.residue_diagnostic.residue),
            status=result.status,
            monodromy=(
                (float(monodromy[0, 0]), float(monodromy[0, 1])),
                (float(monodromy[1, 0]), float(monodromy[1, 1])),
            ),
            trace_m=result.tangent_result.trace_m,
            det_m=result.tangent_result.det_m,
            winding=result.winding,
            radial_label=result.radial_label,
            closure_residual_norm=result.closure_residual_norm,
            min_bphi_over_b=result.min_bphi_over_b,
            residue_scale=self.residue_scale,
            target_weight=target.weight,
            objective_weight=self.objective_weight,
            branch_objective=branch_objective,
            state=result.state,
        )

    def _branch_loss_message(
        self,
        target: RationalTarget,
        key: ResidueBranchKey,
        result: PeriodicOrbitResult,
    ) -> str:
        return (
            "Greene residue objective branch solve requires "
            f"{BRANCH_STATUS_CONVERGED}, got {result.status} for "
            f"target {key[0]!r} branch {key[1]!r}: the RK4 Newton "
            f"iteration from seed section_state "
            f"{tuple(self._branch_state_by_key[key])} settled at "
            f"state {result.state} with winding {result.winding:.6g} "
            f"(expected {target.expected_winding()}), radial_label "
            f"{result.radial_label:.6g} (window {target.radial_window}), "
            f"residue {float(result.residue_diagnostic.residue):.6g}. "
            "This means the supplied residue seeds do not converge to "
            "the requested rational island in the current coil field; "
            "supply seeds validated against this equilibrium/field (a "
            "real fixed-point solve), not synthetic placeholders, drop "
            "--residue-objective-weight for fields without this island, "
            "or set on_branch_loss='treat_as_satisfied' to treat a healed "
            "or drifted island as a satisfied (zero) objective contribution."
        )

    def _compute_gradient(self) -> None:
        if self.objective_weight == 0.0:
            self._dJ = Derivative()
            self._gradient_diagnostics = ()
            return

        if self._branch_values is None:
            self._compute_value()
        lost_branch_keys = self._lost_branch_keys or frozenset()

        derivative = Derivative()
        diagnostics: list[BiotSavartBranchResidueVjpDiagnostic] = []
        for target in self.targets:
            target_id = target.manifest_key()
            for branch in target.branches:
                key = (target_id, branch)
                if key in lost_branch_keys:
                    # Healed/drifted island under treat_as_satisfied: zero
                    # gradient contribution, mirroring the zero J contribution
                    # recorded for this branch in _compute_value.
                    continue
                diagnostic = branch_resolved_biot_savart_residue_vjp(
                    self.biot_savart,
                    self._branch_state_by_key[key],
                    target=target,
                    chart=self.chart,
                    branch=branch,
                    integrator_options=self.integrator_options,
                    solver_options=self.solver_options,
                    r_satisfied=self.r_satisfied,
                    local_difference_step=self.local_difference_step,
                )
                derivative += (
                    self._branch_gradient_scale(target, diagnostic.residue)
                    * diagnostic.derivative
                )
                diagnostics.append(diagnostic)
        self._dJ = derivative
        self._gradient_diagnostics = tuple(diagnostics)

    def _branch_objective(self, target: RationalTarget, residue: float) -> float:
        scaled_residue = float(residue) / self.residue_scale
        return float(
            self.objective_weight
            * 0.5
            * float(target.weight)
            * scaled_residue
            * scaled_residue
        )

    def _branch_gradient_scale(self, target: RationalTarget, residue: float) -> float:
        return float(
            self.objective_weight
            * float(target.weight)
            * float(residue)
            / (self.residue_scale * self.residue_scale)
        )


def residue_branch_seed_from_payload(
    payload: Mapping[str, object],
    *,
    validation_id: str,
    optimizer_taylor_validated: bool | None = None,
    direct_proxy_consistency_validated: bool | None = None,
    real_field_nonzero_winding_validated: bool | None = None,
) -> ResidueBranchSeed:
    seed_validation_id = str(payload.get("validation_id", validation_id))
    _seed_payload_false_only(payload, "optimizer_taylor_validated")
    _seed_payload_false_only(payload, "direct_proxy_consistency_validated")
    _seed_payload_false_only(payload, "real_field_nonzero_winding_validated")
    taylor_validated = (
        False
        if optimizer_taylor_validated is None
        else _seed_bool_value("optimizer_taylor_validated", optimizer_taylor_validated)
    )
    direct_proxy_validated = (
        False
        if direct_proxy_consistency_validated is None
        else _seed_bool_value(
            "direct_proxy_consistency_validated",
            direct_proxy_consistency_validated,
        )
    )
    real_field_validated = (
        False
        if real_field_nonzero_winding_validated is None
        else _seed_bool_value(
            "real_field_nonzero_winding_validated",
            real_field_nonzero_winding_validated,
        )
    )
    return ResidueBranchSeed(
        target_id=str(payload["target_id"]),
        branch=str(payload["branch"]),
        section_state=payload["section_state"],
        validation_id=seed_validation_id,
        optimizer_taylor_validated=taylor_validated,
        direct_proxy_consistency_validated=direct_proxy_validated,
        real_field_nonzero_winding_validated=real_field_validated,
    )


def _seed_payload_false_only(payload: Mapping[str, object], key: str) -> bool:
    value = _seed_bool_value(key, payload.get(key, False))
    if value:
        raise ValueError(
            "Greene residue branch seed validation booleans cannot be "
            f"self-attested in branch_seeds: {key}"
        )
    return False


def _seed_bool_value(key: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Greene residue branch seed {key} must be a boolean")
    return value


def residue_target_from_payload(payload: Mapping[str, object]) -> RationalTarget:
    unknown_keys = sorted(set(payload) - RESIDUE_TARGET_PAYLOAD_KEYS)
    if unknown_keys:
        raise ValueError(f"Unknown Greene residue target keys: {unknown_keys}")
    missing_keys = sorted({"p", "q"} - set(payload))
    if missing_keys:
        raise ValueError(f"Missing Greene residue target keys: {missing_keys}")
    return RationalTarget(
        p=int(payload["p"]),
        q=int(payload["q"]),
        weight=float(payload.get("weight", 1.0)),
        radial_label=payload.get("radial_label"),
        radial_window=payload.get("radial_window"),
        branches=payload.get("branches", ("O", "X")),
        phi0=float(payload.get("phi0", 0.0)),
        nfp=int(payload.get("nfp", 1)),
        convention=str(payload.get("convention", GREENE_IOTA_CONVENTION)),
        fourier_m=payload.get("fourier_m"),
        fourier_n=payload.get("fourier_n"),
        map_convention=str(
            payload.get("map_convention", GREENE_MAP_CONVENTION_FULL_TORUS)
        ),
    )


def load_residue_objective_targets(path: str | Path) -> tuple[RationalTarget, ...]:
    with Path(path).resolve().open(encoding="utf-8") as infile:
        payload = json.load(infile)
    if not isinstance(payload, Mapping):
        raise ValueError("Greene residue targets JSON must be an object")
    targets_payload = payload["targets"]
    if not isinstance(targets_payload, Sequence):
        raise ValueError("Greene residue targets JSON 'targets' must be a sequence")
    return tuple(
        residue_target_from_payload(dict(target)) for target in targets_payload
    )


def load_residue_objective_seeds(
    path: str | Path,
    *,
    target_manifest_id: str,
) -> tuple[str, tuple[ResidueBranchSeed, ...]]:
    with Path(path).resolve().open(encoding="utf-8") as infile:
        payload = json.load(infile)
    if not isinstance(payload, Mapping):
        raise ValueError("Greene residue objective seeds JSON must be an object")
    seed_target_manifest_id = str(payload["target_manifest_id"])
    if seed_target_manifest_id != target_manifest_id:
        raise ValueError(
            "Greene residue objective seeds JSON target_manifest_id does not "
            "match the requested targets"
        )
    validation_status = str(payload["validation_status"])
    if validation_status != GREENE_RESIDUE_OBJECTIVE_VALIDATION_STATUS_PASSED:
        raise ValueError(
            "Greene residue objective seeds JSON validation_status must be passed"
        )
    validation_artifact_id = str(payload["validation_artifact_id"])
    if validation_artifact_id == "":
        raise ValueError(
            "Greene residue objective seeds JSON validation_artifact_id must be nonempty"
        )
    seed_payloads = payload["branch_seeds"]
    if not isinstance(seed_payloads, Sequence):
        raise ValueError(
            "Greene residue objective seeds JSON branch_seeds must be a sequence"
        )
    validation_by_key = _validate_optimizer_taylor_gate(payload, seed_payloads)
    direct_proxy_validation_by_key = _validate_direct_proxy_consistency_gate(
        payload,
        seed_payloads,
        validation_by_key,
    )
    real_field_validation_by_key = _validate_real_field_nonzero_winding_gate(
        payload,
        seed_payloads,
    )
    return (
        validation_artifact_id,
        tuple(
            residue_branch_seed_from_payload(
                dict(seed),
                validation_id=validation_artifact_id,
                optimizer_taylor_validated=(
                    (str(seed["target_id"]), str(seed["branch"])) in validation_by_key
                ),
                direct_proxy_consistency_validated=(
                    (str(seed["target_id"]), str(seed["branch"]))
                    in direct_proxy_validation_by_key
                ),
                real_field_nonzero_winding_validated=(
                    (str(seed["target_id"]), str(seed["branch"]))
                    in real_field_validation_by_key
                ),
            )
            for seed in seed_payloads
        ),
    )


def _seed_state_by_key(
    seed_payloads: Sequence[object],
) -> dict[ResidueBranchKey, SectionState]:
    seed_state_by_key: dict[ResidueBranchKey, SectionState] = {}
    for seed_payload in seed_payloads:
        if not isinstance(seed_payload, Mapping):
            raise ValueError(
                "Greene residue objective branch seed entries must be objects"
            )
        key = (str(seed_payload["target_id"]), str(seed_payload["branch"]))
        if key in seed_state_by_key:
            raise ValueError(
                f"Duplicate Greene residue objective branch seed for {key}"
            )
        seed_state_by_key[key] = _normalize_section_state(seed_payload["section_state"])
    return seed_state_by_key


def _validate_optimizer_taylor_gate(
    payload: Mapping[str, object],
    seed_payloads: Sequence[object],
) -> dict[ResidueBranchKey, Mapping[str, object]]:
    validations_payload = payload.get("optimizer_taylor_validations")
    if not isinstance(validations_payload, Sequence) or isinstance(
        validations_payload, str
    ):
        raise ValueError(
            "Greene residue objective seeds JSON optimizer_taylor_validations "
            "must be a sequence"
        )
    seed_state_by_key = _seed_state_by_key(seed_payloads)
    seed_keys = set(seed_state_by_key)
    validation_by_key: dict[ResidueBranchKey, Mapping[str, object]] = {}
    for validation_payload in validations_payload:
        if not isinstance(validation_payload, Mapping):
            raise ValueError(
                "Greene residue objective optimizer Taylor validation entries "
                "must be objects"
            )
        key = (
            str(validation_payload["target_id"]),
            str(validation_payload["branch"]),
        )
        if key in validation_by_key:
            raise ValueError(
                f"Duplicate Greene residue objective optimizer Taylor validation for {key}"
            )
        validation_by_key[key] = validation_payload
        _validate_optimizer_taylor_diagnostic(
            validation_payload,
            expected_base_state=seed_state_by_key.get(key),
            expected_branch=key[1],
            expected_target_winding=_expected_winding_from_target_id(key[0]),
        )
    missing = sorted(seed_keys - set(validation_by_key))
    if missing:
        raise ValueError(
            f"Missing Greene residue objective optimizer Taylor validations: {missing}"
        )
    unexpected = sorted(set(validation_by_key) - seed_keys)
    if unexpected:
        raise ValueError(
            "Unexpected Greene residue objective optimizer Taylor validations: "
            f"{unexpected}"
        )
    return validation_by_key


def _validate_direct_proxy_consistency_gate(
    payload: Mapping[str, object],
    seed_payloads: Sequence[object],
    optimizer_taylor_validation_by_key: Mapping[ResidueBranchKey, Mapping[str, object]],
) -> dict[ResidueBranchKey, Mapping[str, object]]:
    return _validate_keyed_branch_validation_gate(
        payload,
        seed_payloads,
        payload_key="direct_proxy_consistency_validations",
        validator=lambda validation_payload, key, expected_section_state: (
            _validate_direct_proxy_consistency_proof(
                validation_payload,
                expected_section_state=expected_section_state,
                expected_direct_residue=_optimizer_taylor_base_residue(
                    optimizer_taylor_validation_by_key[key]
                ),
            )
        ),
    )


def _validate_real_field_nonzero_winding_gate(
    payload: Mapping[str, object],
    seed_payloads: Sequence[object],
) -> dict[ResidueBranchKey, Mapping[str, object]]:
    return _validate_keyed_branch_validation_gate(
        payload,
        seed_payloads,
        payload_key="real_field_nonzero_winding_validations",
        validator=_validate_real_field_nonzero_winding_proof,
    )


def _validate_keyed_branch_validation_gate(
    payload: Mapping[str, object],
    seed_payloads: Sequence[object],
    *,
    payload_key: str,
    validator: Callable[[Mapping[str, object], ResidueBranchKey, SectionState], None],
) -> dict[ResidueBranchKey, Mapping[str, object]]:
    validations_payload = payload.get(payload_key, ())
    if not isinstance(validations_payload, Sequence) or isinstance(
        validations_payload,
        str,
    ):
        raise ValueError(
            f"Greene residue objective seeds JSON {payload_key} must be a sequence"
        )
    seed_state_by_key = _seed_state_by_key(seed_payloads)
    seed_keys = set(seed_state_by_key)
    validation_by_key: dict[ResidueBranchKey, Mapping[str, object]] = {}
    for validation_payload in validations_payload:
        if not isinstance(validation_payload, Mapping):
            raise ValueError(
                f"Greene residue objective {payload_key} entries must be objects"
            )
        key = (
            str(validation_payload["target_id"]),
            str(validation_payload["branch"]),
        )
        if key in validation_by_key:
            raise ValueError(
                f"Duplicate Greene residue objective {payload_key} entry for {key}"
            )
        if key not in seed_keys:
            raise ValueError(
                f"Unexpected Greene residue objective {payload_key} entry for {key}"
            )
        validator(validation_payload, key, seed_state_by_key[key])
        validation_by_key[key] = validation_payload
    return validation_by_key


def _expected_winding_from_target_id(target_id: str) -> float:
    for field in str(target_id).split("|"):
        if field.startswith("p="):
            return float(int(field.removeprefix("p=")))
    raise ValueError(
        "Greene residue objective validation target_id must include target winding"
    )


def _optimizer_taylor_base_residue(
    validation_payload: Mapping[str, object],
) -> float:
    diagnostic = validation_payload["diagnostic"]
    if not isinstance(diagnostic, Mapping):
        raise ValueError(
            "Greene residue objective optimizer Taylor validation diagnostic "
            "must be an object"
        )
    return float(diagnostic["base_residue"])


def _validate_proof_section_state(
    validation_payload: Mapping[str, object],
    *,
    expected_section_state: SectionState,
    label: str,
) -> None:
    section_state = _normalize_section_state(validation_payload["section_state"])
    if section_state != expected_section_state:
        raise ValueError(
            f"Greene residue objective {label} validation section_state must "
            "match the branch seed section_state"
        )


def _passed_validation_status(payload: Mapping[str, object], *, label: str) -> None:
    if (
        str(payload["validation_status"])
        != GREENE_RESIDUE_OBJECTIVE_VALIDATION_STATUS_PASSED
    ):
        raise ValueError(
            f"Greene residue objective {label} validation_status must be passed"
        )


def _validate_direct_proxy_consistency_proof(
    validation_payload: Mapping[str, object],
    *,
    expected_section_state: SectionState,
    expected_direct_residue: float,
) -> None:
    _passed_validation_status(
        validation_payload,
        label="direct-vs-proxy consistency",
    )
    _validate_proof_section_state(
        validation_payload,
        expected_section_state=expected_section_state,
        label="direct-vs-proxy consistency",
    )
    branch = str(validation_payload["branch"])
    if branch not in GREENE_BRANCHES:
        raise ValueError(f"Unknown Greene residue branch label: {branch}")
    direct_residue = float(validation_payload["direct_residue"])
    proxy_residue = float(validation_payload["proxy_residue"])
    absolute_residue_difference = float(
        validation_payload["absolute_residue_difference"]
    )
    residue_difference_tolerance = float(
        validation_payload["residue_difference_tolerance"]
    )
    expected_difference = abs(direct_residue - proxy_residue)
    if (
        not isfinite(direct_residue)
        or not isfinite(proxy_residue)
        or not isfinite(absolute_residue_difference)
        or not isfinite(residue_difference_tolerance)
        or not isfinite(expected_direct_residue)
        or residue_difference_tolerance < 0.0
    ):
        raise ValueError(
            "Greene residue objective direct-vs-proxy validation residues "
            "and tolerance must be finite"
        )
    if not np.isclose(
        absolute_residue_difference,
        expected_difference,
        rtol=GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_RESIDUAL_RTOL,
        atol=GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_RESIDUAL_ATOL,
    ):
        raise ValueError(
            "Greene residue objective direct-vs-proxy validation difference "
            "must equal abs(direct_residue - proxy_residue)"
        )
    if not np.isclose(
        direct_residue,
        expected_direct_residue,
        rtol=GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_RESIDUAL_RTOL,
        atol=GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_RESIDUAL_ATOL,
    ):
        raise ValueError(
            "Greene residue objective direct-vs-proxy validation direct_residue "
            "must match the optimizer Taylor base_residue"
        )
    if absolute_residue_difference > residue_difference_tolerance:
        raise ValueError(
            "Greene residue objective direct-vs-proxy validation difference "
            "exceeds tolerance"
        )


def _validate_real_field_nonzero_winding_proof(
    validation_payload: Mapping[str, object],
    key: ResidueBranchKey,
    expected_section_state: SectionState,
) -> None:
    _passed_validation_status(
        validation_payload,
        label="real-field nonzero-winding",
    )
    _validate_proof_section_state(
        validation_payload,
        expected_section_state=expected_section_state,
        label="real-field nonzero-winding",
    )
    branch = str(validation_payload["branch"])
    if branch not in GREENE_BRANCHES:
        raise ValueError(f"Unknown Greene residue branch label: {branch}")
    if str(validation_payload["branch_status"]) != BRANCH_STATUS_CONVERGED:
        raise ValueError(
            "Greene residue objective real-field nonzero-winding validation "
            "branch_status must be converged"
        )
    expected_winding = float(validation_payload["expected_winding"])
    target_expected_winding = _expected_winding_from_target_id(key[0])
    observed_winding = float(validation_payload["observed_winding"])
    winding_residual = float(validation_payload["winding_residual"])
    winding_tolerance = float(validation_payload["winding_tolerance"])
    if (
        expected_winding == 0.0
        or not isfinite(expected_winding)
        or not isfinite(observed_winding)
        or not isfinite(winding_residual)
        or not isfinite(winding_tolerance)
        or winding_tolerance < 0.0
    ):
        raise ValueError(
            "Greene residue objective real-field nonzero-winding validation "
            "requires finite nonzero expected winding and finite tolerance"
        )
    if expected_winding != target_expected_winding:
        raise ValueError(
            "Greene residue objective real-field nonzero-winding validation "
            "expected_winding must match the target winding"
        )
    if not np.isclose(
        winding_residual,
        observed_winding - expected_winding,
        rtol=GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_RESIDUAL_RTOL,
        atol=GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_WINDING_ATOL,
    ):
        raise ValueError(
            "Greene residue objective real-field nonzero-winding validation "
            "residual must equal observed_winding minus expected_winding"
        )
    if abs(winding_residual) > winding_tolerance:
        raise ValueError(
            "Greene residue objective real-field nonzero-winding validation "
            "residual exceeds tolerance"
        )


def _validate_optimizer_taylor_diagnostic(
    validation_payload: Mapping[str, object],
    *,
    expected_base_state: SectionState | None,
    expected_branch: str,
    expected_target_winding: float,
) -> None:
    diagnostic = validation_payload["diagnostic"]
    if not isinstance(diagnostic, Mapping):
        raise ValueError(
            "Greene residue objective optimizer Taylor validation diagnostic "
            "must be an object"
        )
    if str(diagnostic["mode"]) != BIOT_SAVART_BRANCH_RESOLVED_VJP_TAYLOR_MODE:
        raise ValueError(
            "Greene residue objective optimizer Taylor validation mode must be "
            f"{BIOT_SAVART_BRANCH_RESOLVED_VJP_TAYLOR_MODE}"
        )
    if str(diagnostic["base_status"]) != BRANCH_STATUS_CONVERGED:
        raise ValueError(
            "Greene residue objective optimizer Taylor validation base_status "
            "must be converged"
        )
    branch = str(diagnostic["branch"])
    if branch != expected_branch:
        raise ValueError(
            "Greene residue objective optimizer Taylor validation branch must "
            "match the branch seed"
        )
    expected_winding = float(diagnostic["expected_winding"])
    base_winding = float(diagnostic["base_winding"])
    base_winding_residual = float(diagnostic["base_winding_residual"])
    base_raw_return_section_winding = float(
        diagnostic["base_raw_return_section_winding"]
    )
    base_det_m = float(diagnostic["base_det_m"])
    base_residue_classification = str(diagnostic["base_residue_classification"])
    if (
        not isfinite(expected_winding)
        or not isfinite(base_winding)
        or not isfinite(base_winding_residual)
        or not isfinite(base_raw_return_section_winding)
        or not isfinite(base_det_m)
        or base_residue_classification == ""
    ):
        raise ValueError(
            "Greene residue objective optimizer Taylor validation branch and "
            "winding diagnostics must be finite"
        )
    if expected_winding != expected_target_winding:
        raise ValueError(
            "Greene residue objective optimizer Taylor validation expected_winding "
            "must match the target winding"
        )
    if (
        abs(base_winding - expected_winding)
        > GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_WINDING_ATOL
        or abs(base_winding_residual)
        > GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_WINDING_ATOL
    ):
        raise ValueError(
            "Greene residue objective optimizer Taylor validation base winding "
            "must match the requested target winding"
        )
    base_state = _normalize_section_state(diagnostic["base_state"])
    if expected_base_state is not None and base_state != expected_base_state:
        raise ValueError(
            "Greene residue objective optimizer Taylor validation base_state "
            "must match the branch seed section_state"
        )
    direction_norm = float(diagnostic["direction_norm"])
    if not isfinite(direction_norm) or direction_norm <= 0.0:
        raise ValueError(
            "Greene residue objective optimizer Taylor validation direction_norm "
            "must be positive"
        )
    samples = diagnostic["samples"]
    if not isinstance(samples, Sequence) or isinstance(samples, str):
        raise ValueError(
            "Greene residue objective optimizer Taylor validation samples "
            "must be a sequence"
        )
    if len(samples) < GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_MIN_SAMPLES:
        raise ValueError(
            "Greene residue objective optimizer Taylor validation requires at "
            f"least {GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_MIN_SAMPLES} samples"
        )
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError(
                "Greene residue objective optimizer Taylor validation samples "
                "must be objects"
            )
        if str(sample["status"]) != BRANCH_STATUS_CONVERGED:
            raise ValueError(
                "Greene residue objective optimizer Taylor validation sample "
                "status must be converged"
            )
        if str(sample["branch"]) != expected_branch:
            raise ValueError(
                "Greene residue objective optimizer Taylor validation sample "
                "branch must match the branch seed"
            )
        step = float(sample["step"])
        residue = float(sample["residue"])
        first_order_prediction = float(sample["first_order_prediction"])
        residual = float(sample["residual"])
        absolute_residual = float(sample["absolute_residual"])
        winding = float(sample["winding"])
        winding_residual = float(sample["winding_residual"])
        raw_return_section_winding = float(sample["raw_return_section_winding"])
        det_m = float(sample["det_m"])
        residue_classification = str(sample["residue_classification"])
        expected_residual = residue - first_order_prediction
        if (
            step <= 0.0
            or not isfinite(step)
            or not isfinite(residue)
            or not isfinite(first_order_prediction)
            or not isfinite(residual)
            or absolute_residual <= 0.0
            or not isfinite(absolute_residual)
            or not isfinite(winding)
            or not isfinite(winding_residual)
            or not isfinite(raw_return_section_winding)
            or not isfinite(det_m)
            or residue_classification == ""
        ):
            raise ValueError(
                "Greene residue objective optimizer Taylor validation samples "
                "must have finite positive steps, finite positive residuals, "
                "and finite branch/winding diagnostics"
            )
        if (
            abs(winding - expected_winding)
            > GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_WINDING_ATOL
            or abs(winding_residual) > GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_WINDING_ATOL
        ):
            raise ValueError(
                "Greene residue objective optimizer Taylor validation sample "
                "winding must match the requested target winding"
            )
        if not np.isclose(
            residual,
            expected_residual,
            rtol=GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_RESIDUAL_RTOL,
            atol=GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_RESIDUAL_ATOL,
        ):
            raise ValueError(
                "Greene residue objective optimizer Taylor validation sample "
                "residual must equal residue minus first_order_prediction"
            )
        if not np.isclose(
            absolute_residual,
            abs(residual),
            rtol=GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_RESIDUAL_RTOL,
            atol=GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_RESIDUAL_ATOL,
        ):
            raise ValueError(
                "Greene residue objective optimizer Taylor validation sample "
                "absolute_residual must equal abs(residual)"
            )
    observed_orders = diagnostic["observed_orders"]
    if not isinstance(observed_orders, Sequence) or isinstance(observed_orders, str):
        raise ValueError(
            "Greene residue objective optimizer Taylor validation observed_orders "
            "must be a sequence"
        )
    if len(observed_orders) != len(samples) - 1:
        raise ValueError(
            "Greene residue objective optimizer Taylor validation observed_orders "
            "must cover the sample window"
        )
    computed_orders = []
    for previous, current in zip(samples[:-1], samples[1:], strict=True):
        previous_step = float(previous["step"])
        current_step = float(current["step"])
        previous_residual = float(previous["absolute_residual"])
        current_residual = float(current["absolute_residual"])
        computed_orders.append(
            float(
                np.log(previous_residual / current_residual)
                / np.log(previous_step / current_step)
            )
        )
    claimed_orders = [float(order) for order in observed_orders]
    if not np.allclose(
        claimed_orders,
        computed_orders,
        rtol=GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_ORDER_RTOL,
        atol=GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_ORDER_ATOL,
    ):
        raise ValueError(
            "Greene residue objective optimizer Taylor validation observed_orders "
            "must match the sample residuals"
        )
    minimum_order = min(claimed_orders)
    if (
        not isfinite(minimum_order)
        or minimum_order < GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_MIN_ORDER
    ):
        raise ValueError(
            "Greene residue objective optimizer Taylor validation minimum order "
            f"must be at least {GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_MIN_ORDER}"
        )


def residue_objective_target_manifest_id(targets: Sequence[RationalTarget]) -> str:
    payload = [target.manifest_key() for target in targets]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=False).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def disabled_residue_objective_payload(
    *,
    target_manifest_id: str,
    objective_weight: float,
    residue_scale: float,
) -> dict[str, object]:
    return {
        "schema_version": GREENE_RESIDUE_OBJECTIVE_SCHEMA_VERSION,
        "enabled": False,
        "target_manifest_id": str(target_manifest_id),
        "validation_id": "",
        "validation_evidence_kind": GREENE_RESIDUE_OBJECTIVE_VALIDATION_EVIDENCE_KIND,
        "validation_limitations": list(GREENE_RESIDUE_OBJECTIVE_VALIDATION_LIMITATIONS),
        "dof_gradient_method": BIOT_SAVART_BRANCH_RESIDUE_DOF_GRADIENT_METHOD,
        "local_sensitivity_method": (
            BIOT_SAVART_BRANCH_RESIDUE_LOCAL_SENSITIVITY_METHOD
        ),
        "local_sensitivity_limitations": list(
            BIOT_SAVART_BRANCH_RESIDUE_LOCAL_SENSITIVITY_LIMITATIONS
        ),
        "objective_weight": float(objective_weight),
        "residue_scale": float(residue_scale),
        "value": 0.0,
        "gradient_norm": 0.0,
        "branches": [],
    }


def _validated_seed_state_map(
    targets: Sequence[RationalTarget],
    branch_seeds: Sequence[ResidueBranchSeed],
    *,
    validation_id: str,
    require_optimizer_taylor: bool,
) -> dict[ResidueBranchKey, SectionState]:
    expected_keys = _target_branch_keys(targets)
    if len(set(expected_keys)) != len(expected_keys):
        raise ValueError("Duplicate Greene residue objective target/branch keys")
    target_by_key = {
        (target.manifest_key(), branch): target
        for target in targets
        for branch in target.branches
    }
    seed_by_key: dict[ResidueBranchKey, ResidueBranchSeed] = {}
    for seed in branch_seeds:
        if seed.validation_id != validation_id:
            raise ValueError(
                "Greene residue objective seed validation_id does not match "
                f"{validation_id!r}"
            )
        if require_optimizer_taylor and not seed.optimizer_taylor_validated:
            raise ValueError(
                "Greene residue objective branch seeds require optimizer Taylor "
                "validation before nonzero objective use"
            )
        if require_optimizer_taylor and not seed.direct_proxy_consistency_validated:
            raise ValueError(
                "Greene residue objective branch seeds require direct-vs-proxy "
                "physics validation before nonzero objective use"
            )
        target = target_by_key.get(seed.key)
        if (
            require_optimizer_taylor
            and target is not None
            and target.expected_winding() != 0
            and not seed.real_field_nonzero_winding_validated
        ):
            raise ValueError(
                "Greene residue objective nonzero-winding branch seeds require "
                "real-field convergence validation before nonzero objective use"
            )
        if seed.key in seed_by_key:
            raise ValueError(
                f"Duplicate Greene residue objective branch seed for {seed.key}"
            )
        seed_by_key[seed.key] = seed

    missing = [key for key in expected_keys if key not in seed_by_key]
    if missing:
        raise ValueError(f"Missing Greene residue objective branch seeds: {missing}")
    unexpected = sorted(set(seed_by_key) - set(expected_keys))
    if unexpected:
        raise ValueError(
            f"Unknown Greene residue objective branch seed targets: {unexpected}"
        )
    return {key: seed_by_key[key].section_state for key in expected_keys}


def _target_branch_keys(
    targets: Sequence[RationalTarget],
) -> tuple[ResidueBranchKey, ...]:
    return tuple(
        (target.manifest_key(), branch)
        for target in targets
        for branch in target.branches
    )


def _normalize_section_state(state: Sequence[float]) -> SectionState:
    if len(state) != 2:
        raise ValueError("Greene residue branch seed section_state must have length 2")
    radius = float(state[0])
    height = float(state[1])
    if not isfinite(radius) or not isfinite(height):
        raise ValueError("Greene residue branch seed section_state must be finite")
    if radius <= 0.0:
        raise ValueError("Greene residue branch seed section_state requires R > 0")
    return (radius, height)
