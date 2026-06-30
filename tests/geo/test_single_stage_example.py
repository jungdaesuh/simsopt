import copy
import importlib.util
from contextlib import ExitStack, contextmanager
import io
import json
import os
import re
import sys
import tempfile
import unittest
import uuid
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import simsopt.geo.surfaceobjectives as surfaceobjectives_module
from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable
from examples.single_stage_optimization.banana_opt.topology.kam_birkhoff import (
    KAM_FRACTION_SEMANTICS,
)
from examples.single_stage_optimization.STAGE_2 import (
    banana_coil_solver as stage2_solver,
)
from simsopt.field.coil import Current, ScaledCurrent
from simsopt.objectives.utilities import forward_backward


EXAMPLE_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "SINGLE_STAGE"
    / "single_stage_banana_example.py"
)
SIGNED_CW_WOUT_PATH = (
    Path(__file__).resolve().parents[1] / "test_files" / "wout_10x10.nc"
)
POSITIVE_CCW_WOUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "test_files"
    / "wout_LandremanPaul2021_QA_lowres.nc"
)
BOOZER_SURFACE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "simsopt" / "geo" / "boozersurface.py"
)
TOPOLOGY_SCORER_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "topology_scorer.py"
)
POINCARE_SURFACES_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "POINCARE_PLOTTING"
    / "poincare_surfaces.py"
)
TOPOLOGY_FIDELITY_LADDER_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "banana_opt"
    / "topology_fidelity_ladder.py"
)
ALM_UTILS_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "alm_utils.py"
)
HARDWARE_CONTRACTS_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "banana_opt"
    / "hardware_contracts.py"
)
WORKFLOW_HELPERS_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "workflow_helpers.py"
)
WORKFLOW_RUNNER_COMMON_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "workflow_runner_common.py"
)
TEST_MPOL = 8
TEST_NTOR = 6
TEST_VOL_TARGET = 0.1
TEST_IOTA = 0.15
TEST_G0 = 1.0
TEST_BOOZER_I = 0.37


def _load_module_from_path(module_path, name_prefix, *, register_in_sys_modules=False):
    spec = importlib.util.spec_from_file_location(
        f"{name_prefix}_{uuid.uuid4().hex}",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_single_stage_example_module():
    return _load_module_from_path(
        EXAMPLE_MODULE_PATH,
        "single_stage_banana_example",
    )


def load_topology_scorer_module():
    return _load_module_from_path(
        TOPOLOGY_SCORER_MODULE_PATH,
        "topology_scorer",
    )


def load_poincare_surfaces_module():
    return _load_module_from_path(
        POINCARE_SURFACES_MODULE_PATH,
        "poincare_surfaces",
    )


def load_topology_fidelity_ladder_module():
    # The ladder module uses package-relative imports (from .topology.kam_birkhoff
    # import ...), so the fresh copy must be loaded under a dotted name inside the
    # real banana_opt package for the relative import to resolve.
    return _load_module_from_path(
        TOPOLOGY_FIDELITY_LADDER_MODULE_PATH,
        "examples.single_stage_optimization.banana_opt.topology_fidelity_ladder",
        register_in_sys_modules=True,
    )


def load_alm_utils_module():
    return _load_module_from_path(
        ALM_UTILS_MODULE_PATH,
        "alm_utils",
    )


def load_hardware_contracts_module():
    return _load_module_from_path(
        HARDWARE_CONTRACTS_MODULE_PATH,
        "hardware_contracts",
    )


def in_bounds_lcfs_major_radius_m():
    hardware_contracts = load_hardware_contracts_module()
    return hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M - 0.01


def in_bounds_lcfs_minor_radius_m():
    hardware_contracts = load_hardware_contracts_module()
    return hardware_contracts.TARGET_LCFS_MAX_MINOR_RADIUS_M - 0.01


def diagnostic_search_eval_payload(base_payload):
    return {
        "J_QS": 2.5e-4,
        "dJ_QS": np.array([0.1, -0.1]),
        "J_Boozer": 4.0e-7,
        "dJ_Boozer": np.array([0.2, -0.2]),
        "J_iota": 1.0e-3,
        "dJ_iota": np.array([0.3, -0.3]),
        "J_curvature": 0.0,
        "dJ_curvature": np.array([0.0, 0.0]),
        **base_payload,
        "diagnostics_included": True,
    }


def topology_hardware_snapshot(
    *,
    search_success=True,
    artifact_success=True,
    search_violations=None,
    artifact_violations=None,
):
    def hardware_status(success, violations):
        resolved_violations = [] if violations is None else list(violations)
        return {
            "success": bool(success),
            "violations": resolved_violations,
            "constraints": {
                "max_curvature": {
                    "success": bool(success),
                    "value": 99.0 if success else 102.0,
                    "threshold": 100.0,
                    "violation": 0.0 if success else 2.0,
                }
            },
            "violation_ratios": {
                "max_curvature_penalty": 0.0 if success else 0.02,
            },
        }

    return {
        "search_hardware_status": hardware_status(
            search_success,
            search_violations,
        ),
        "artifact_hardware_status": hardware_status(
            artifact_success,
            artifact_violations,
        ),
    }


def search_hardware_penalty_payload(values=(0.0, 0.0, 0.0)):
    return {
        "constraint_names": [
            "coil_coil_spacing",
            "coil_surface_spacing",
            "max_curvature",
        ],
        "dual_update_values": np.asarray(values, dtype=float),
        "feasibility_values": np.asarray(values, dtype=float).copy(),
        "search_hardware_constraint_payload_kind": "penalty_objective",
    }


def seed_near_miss_diagnostic_globals(module, *, coil_length=0.44):
    module.curvelength = SimpleNamespace(J=lambda: coil_length)
    module.CurveLength = lambda curve: SimpleNamespace(J=lambda: coil_length)
    module.VV = None
    module.OUT_DIR_ITER = ""


def load_workflow_helpers_module():
    return _load_module_from_path(
        WORKFLOW_HELPERS_MODULE_PATH,
        "workflow_helpers",
        register_in_sys_modules=True,
    )


def load_workflow_runner_common_module():
    return _load_module_from_path(
        WORKFLOW_RUNNER_COMMON_MODULE_PATH,
        "workflow_runner_common",
        register_in_sys_modules=True,
    )


def _mock_topology_score_result(
    *,
    stop_reason,
    first_exit_time,
    survival_fraction=0.5,
    survived_lines=1,
    seed_mode="midplane_radial_sweep",
    field_mode="native",
):
    return {
        "survival_fraction": float(survival_fraction),
        "survived_lines": int(survived_lines),
        "stop_reason_counts": {str(stop_reason): 1},
        "first_exit": {
            "first_exit_time": float(first_exit_time),
            "first_exit_angle": 0.0,
            "stop_reason": str(stop_reason),
        },
        "seed_contract": {"mode": str(seed_mode)},
        "field_model": {"selected_mode": str(field_mode)},
    }


def phase1_runtime_kwargs(module, *, phase1_config=None):
    resolved_phase1_config = (
        module.build_phase1_config() if phase1_config is None else phase1_config
    )
    return {
        "phase1_config": resolved_phase1_config,
        "refinement_eligible_fn": module.refinement_eligible_incumbent,
        "repair_progress_state_fn": module.repair_progress_state,
    }


def build_banana_current_report_fixture(module, *, seed_currents_A=(1.5e4, -1.5e4)):
    current_a = Current(1.5e4)
    current_b = Current(-1.5e4)
    banana_current_state = module.SingleStageBananaCurrentState(
        mode="independent",
        currents=(current_a, current_b),
        seed_currents_A=seed_currents_A,
    )
    objective = SimpleNamespace(
        dof_names=("geom:0", *current_a.dof_names, "geom:1", *current_b.dof_names),
        lower_bounds=np.array([-1.0, -1.6e4, -1.0, -1.6e4], dtype=float),
        upper_bounds=np.array([1.0, 1.6e4, 1.0, 1.6e4], dtype=float),
    )
    return current_a, current_b, banana_current_state, objective


def make_frontier_goal_config(module, **overrides):
    config = {
        "iota_reference": 0.10,
        "iota_scale": 0.05,
        "volume_reference": 0.10,
        "volume_scale": 0.01,
        "qs_reference": 1.0e-4,
        "qs_scale": 2.5e-5,
        "boozer_reference": 1.0e-6,
        "boozer_scale": 2.5e-7,
        "boozer_trust_threshold": 1.0e-5,
        "boozer_trust_penalty_scale": 5.0e-5,
        "effective_qs_weight": 1.0,
        "effective_boozer_weight": 1.0,
        "effective_iota_weight": 1.0,
        "effective_volume_weight": 1.0,
        "scalarization_type": "weight_schedule_v1",
        "chebyshev_rho": 1.0e-3,
        "chebyshev_sharpness": 12.0,
        "chebyshev_weight_iota": 1.0,
        "chebyshev_weight_volume": 1.0,
        "chebyshev_weight_qa": 1.0,
        "chebyshev_weight_boozer": 1.0,
        "epsilon_constraint_qa_max": None,
        "epsilon_constraint_boozer_max": None,
        "epsilon_penalty_weight": 4.0,
    }
    config.update(overrides)
    return module.FrontierGoalConfig(**config)


class FakeSurfPrev:
    def __init__(self):
        self.nfp = 5
        self.quadpoints_phi = np.linspace(0, 1 / self.nfp, 13, endpoint=False)
        self.quadpoints_theta = np.linspace(0, 1, 17, endpoint=False)

    def gamma(self):
        return np.zeros((self.quadpoints_phi.size, self.quadpoints_theta.size, 3))


class FakeSurfaceXYZTensorFourier:
    instances = []

    def __init__(
        self,
        *,
        mpol,
        ntor,
        nfp,
        stellsym,
        quadpoints_theta,
        quadpoints_phi,
        dofs=None,
    ):
        self.mpol = mpol
        self.ntor = ntor
        self.nfp = nfp
        self.stellsym = stellsym
        self.quadpoints_theta = np.asarray(quadpoints_theta)
        self.quadpoints_phi = np.asarray(quadpoints_phi)
        self.dofs = np.array([1.0]) if dofs is None else np.asarray(dofs)
        FakeSurfaceXYZTensorFourier.instances.append(self)

    def least_squares_fit(self, gamma):
        self.fitted_gamma = gamma

    def is_self_intersecting(self):
        return False

    def volume(self):
        return 1.0


class FakeVolume:
    def __init__(self, surface):
        self.surface = surface

    def J(self):
        return self.surface.volume()


class FakeBoozerSurface:
    def __init__(
        self, bs, surface, label, targetlabel, constraint_weight, options=None, I=0.0
    ):
        self.bs = bs
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel
        self.constraint_weight = constraint_weight
        self.options = options or {}
        self.I = I
        self.res = {"success": True, "iota": 0.15, "G": 1.0, "I": I}
        self.need_to_run_code = True

    def run_code(self, iota, G):
        self.need_to_run_code = False
        return self.res


class FakeResolvedSurface:
    def __init__(
        self,
        *,
        mpol,
        ntor,
        stellsym,
        nfp,
        quadpoints_phi,
        quadpoints_theta,
    ):
        self.mpol = mpol
        self.ntor = ntor
        self.stellsym = stellsym
        self.nfp = nfp
        self.quadpoints_phi = np.asarray(quadpoints_phi)
        self.quadpoints_theta = np.asarray(quadpoints_theta)
        self._dofs = np.zeros(2)

    def set_dofs(self, dofs):
        self._dofs = np.asarray(dofs, dtype=float)

    def get_dofs(self):
        return self._dofs.copy()

    def gamma(self):
        return np.zeros((self.quadpoints_phi.size, self.quadpoints_theta.size, 3))


class FakeLabel:
    def J(self):
        return 0.0

    def dJ_by_dsurfacecoefficients(self):
        return np.zeros(2)


class FakeObjectiveBiotSavart:
    def __init__(self):
        self.points = None
        self.last_vjp_input = None

    def set_points(self, points):
        self.points = np.asarray(points, dtype=float)

    def B_vjp(self, dJ_by_dB):
        self.last_vjp_input = np.asarray(dJ_by_dB, dtype=float)
        return np.array([3.0, -1.0])


class FakeDifferentiableBiotSavart(Optimizable):
    def __init__(self, x0):
        super().__init__(x0=np.asarray(x0, dtype=float))
        self.points = None
        self.b_and_dB_vjp_calls = 0

    def set_points(self, points):
        self.points = np.asarray(points, dtype=float)

    @staticmethod
    def weights(num_components):
        idx = np.arange(num_components, dtype=float)
        return np.column_stack(
            (
                0.25 + 0.01 * idx,
                -0.15 + 0.02 * ((idx % 5.0) - 2.0),
            )
        )

    def residual_vector(self, surface, *, weight_inv_modB, current_I):
        num_components = 3 * surface.quadpoints_phi.size * surface.quadpoints_theta.size
        offset = 0.03 * current_I + (0.07 if weight_inv_modB else -0.05)
        return self.weights(num_components) @ self.x + offset

    def B_vjp(self, dJ_by_dB):
        flat_dJ_by_dB = np.asarray(dJ_by_dB, dtype=float).reshape(-1)
        return Derivative({self: self.weights(flat_dJ_by_dB.size).T @ flat_dJ_by_dB})

    def B_and_dB_vjp(self, v, vgrad):
        del v, vgrad
        self.b_and_dB_vjp_calls += 1
        zero = Derivative({self: np.zeros_like(self.x)})
        return zero, zero


class FakeDifferentiableLabel:
    def __init__(self, *, value=0.4, derivative=(0.2, -0.1)):
        self.value = float(value)
        self.derivative = np.asarray(derivative, dtype=float)

    def J(self):
        return self.value

    def dJ_by_dsurfacecoefficients(self):
        return self.derivative.copy()


class FakeParentBoozerSurface:
    def __init__(
        self,
        *,
        surface,
        label,
        targetlabel,
        need_to_run_code,
        res,
        constraint_weight=1.0,
    ):
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel
        self.need_to_run_code = need_to_run_code
        self.res = res
        self.constraint_weight = constraint_weight
        self.ancestors = []
        self.name = "FakeBoozerSurface"
        self.dofs = object()
        self.local_full_dof_size = 0
        self.local_dof_size = 0
        self._id = SimpleNamespace(id=0)

    def _add_child(self, child):
        del child


class FakeDifferentiableBoozerSurface(Optimizable):
    def __init__(
        self,
        *,
        surface,
        biotsavart,
        current_I,
        weight_inv_modB,
        implicit_scale=(0.0, 0.0),
        res_type="exact",
    ):
        super().__init__(depends_on=[biotsavart])
        nsurfdofs = surface.get_dofs().size
        self.surface = surface
        self.biotsavart = biotsavart
        self.label = FakeDifferentiableLabel()
        self.targetlabel = 0.1
        self.constraint_weight = 1.3
        self.need_to_run_code = False
        self.implicit_scale = np.asarray(implicit_scale, dtype=float)
        self.res = {
            "iota": -0.4,
            "G": 1.2,
            "I": float(current_I),
            "type": res_type,
            "weight_inv_modB": bool(weight_inv_modB),
            "PLU": (
                np.eye(nsurfdofs + 2),
                np.eye(nsurfdofs + 2),
                np.eye(nsurfdofs + 2),
            ),
            "vjp": self._vjp,
        }

    def run_code(self, iota, G):
        del iota, G
        return self.res

    def run_code_from_last_solution(self):
        return self.run_code(self.res["iota"], self.res["G"])

    def _vjp(self, adj, booz_surf, iota, G):
        del booz_surf, iota, G
        return Derivative({self.biotsavart: self.implicit_scale * float(np.sum(adj))})


class FakeAlgebraicObjective:
    def __init__(self, value, gradient):
        self._value = float(value)
        self._gradient = np.asarray(gradient, dtype=float)

    def J(self):
        return self._value

    def dJ(self):
        return self._gradient.copy()

    def __add__(self, other):
        if other == 0:
            return self
        return FakeAlgebraicObjective(
            self._value + other._value, self._gradient + other._gradient
        )

    __radd__ = __add__

    def __mul__(self, scalar):
        return FakeAlgebraicObjective(self._value * scalar, self._gradient * scalar)

    __rmul__ = __mul__


class FakeProjectedObjective(FakeAlgebraicObjective):
    def __init__(self, value, gradient, projected_gradient):
        super().__init__(value, gradient)
        self._projected_gradient = np.asarray(projected_gradient, dtype=float)

    def dJ(self, partials=False):
        if not partials:
            return super().dJ()
        return lambda objective_optimizable: self._projected_gradient.copy()

    def __add__(self, other):
        if other == 0:
            return self
        return FakeProjectedObjective(
            self._value + other._value,
            self._gradient + other._gradient,
            self._projected_gradient + other._projected_gradient,
        )

    __radd__ = __add__

    def __mul__(self, scalar):
        return FakeProjectedObjective(
            self._value * scalar,
            self._gradient * scalar,
            self._projected_gradient * scalar,
        )

    __rmul__ = __mul__


class FakeResidueObjective(FakeAlgebraicObjective):
    def to_json_dict(self):
        return {
            "schema_version": "test_residue_objective_v1",
            "enabled": True,
            "target_manifest_id": "test-targets",
            "validation_id": "test-validation",
            "objective_weight": 1.0,
            "residue_scale": 1.0,
            "value": self.J(),
            "branches": [],
        }


class SingleStageExampleTests(unittest.TestCase):
    def setUp(self):
        FakeSurfaceXYZTensorFourier.instances = []

    def load_module(self):
        return load_single_stage_example_module()

    def initialize_boozer_surface(
        self,
        module,
        surf_prev,
        *,
        constraint_weight,
        initial_surface_guess=None,
    ):
        with (
            patch.object(
                module, "SurfaceXYZTensorFourier", FakeSurfaceXYZTensorFourier
            ),
            patch.object(module, "Volume", FakeVolume),
        ):
            return module.initialize_boozer_surface(
                surf_prev,
                mpol=TEST_MPOL,
                ntor=TEST_NTOR,
                bs=object(),
                vol_target=TEST_VOL_TARGET,
                constraint_weight=constraint_weight,
                iota=TEST_IOTA,
                G0=TEST_G0,
                boozer_I=TEST_BOOZER_I,
                initial_surface_guess=initial_surface_guess,
                boozer_surface_cls=FakeBoozerSurface,
            )

    def residual_module(self, module):
        return sys.modules[module.BoozerResidualExact.__module__]

    @contextmanager
    def patched_boozer_objective_term_types(self, module):
        constructed_biot_savarts = []

        class _BiotSavart:
            def __init__(self, coils):
                self.coils = coils
                constructed_biot_savarts.append(self)

        class _Iotas:
            def __init__(self, boozer_surface):
                self.boozer_surface = boozer_surface

        class _NonQS:
            def __init__(self, boozer_surface, biotsavart):
                self.boozer_surface = boozer_surface
                self.biotsavart = biotsavart

        class _Residual:
            def __init__(
                self,
                boozer_surface,
                biotsavart,
                *,
                include_label_constraint=True,
                threshold=0.0,
            ):
                self.boozer_surface = boozer_surface
                self.biotsavart = biotsavart
                self.include_label_constraint = include_label_constraint
                self.threshold = threshold

        class _ExactResidual(_Residual):
            pass

        with (
            patch.object(module, "BiotSavart", _BiotSavart),
            patch.object(
                module,
                "Iotas",
                _Iotas,
            ),
            patch.object(module, "NonQuasiSymmetricRatio", _NonQS),
            patch.object(
                module,
                "RefinedBoozerResidual",
                _Residual,
            ),
            patch.object(
                module,
                "BoozerResidualExact",
                _ExactResidual,
            ),
        ):
            yield SimpleNamespace(
                biot_savarts=constructed_biot_savarts,
                residual_cls=_Residual,
                exact_residual_cls=_ExactResidual,
            )

    def test_boozer_derived_objective_terms_share_one_biotsavart_per_surface(self):
        module = self.load_module()
        surface_data = [
            {"boozer_surface": SimpleNamespace(constraint_weight=1.0)},
            {"boozer_surface": SimpleNamespace(constraint_weight=1.0)},
        ]

        with self.patched_boozer_objective_term_types(module) as patched_types:
            search_terms = module.build_boozer_derived_objective_terms(
                "search",
                surface_data,
                ["coil"],
                boozer_residual_threshold=1.0e-4,
            )
            final_terms = module.build_boozer_derived_objective_terms(
                "final",
                surface_data,
                ["coil"],
            )

        self.assertEqual(len(search_terms["boozer_objective_biot_savarts"]), 2)
        self.assertIs(
            search_terms["nonQSs"][0].biotsavart,
            search_terms["brs"][0].biotsavart,
        )
        self.assertIs(
            search_terms["nonQSs"][1].biotsavart,
            search_terms["brs"][1].biotsavart,
        )
        self.assertIsNot(
            search_terms["brs"][0].biotsavart,
            search_terms["brs"][1].biotsavart,
        )
        self.assertIsInstance(search_terms["brs"][0], patched_types.residual_cls)
        self.assertNotIsInstance(
            search_terms["brs"][0], patched_types.exact_residual_cls
        )
        self.assertIsInstance(final_terms["brs"][0], patched_types.exact_residual_cls)
        self.assertEqual(search_terms["brs"][0].threshold, 1.0e-4)
        self.assertEqual(search_terms["brs"][1].threshold, 1.0e-4)
        self.assertEqual(len(patched_types.biot_savarts), 4)

    def test_boozer_residual_stage_selection_keeps_exact_residual_final_only(self):
        module = self.load_module()

        with self.patched_boozer_objective_term_types(module) as patched_types:
            self.assertIs(
                module.boozer_residual_class_for_stage("initial"),
                patched_types.residual_cls,
            )
            self.assertIs(
                module.boozer_residual_class_for_stage("search"),
                patched_types.residual_cls,
            )
            self.assertIs(
                module.boozer_residual_class_for_stage("final"),
                patched_types.exact_residual_cls,
            )

    def test_boozer_residual_threshold_for_stage_skips_final_and_thresholded_physics(
        self,
    ):
        module = self.load_module()

        cases = (
            ("search", "alm", "weighted_sum", 1.0e-4),
            ("final", "alm", "weighted_sum", 0.0),
            ("search", "alm", "thresholded_physics", 0.0),
        )
        for stage, constraint_method, alm_formulation, expected in cases:
            with self.subTest(stage=stage, alm_formulation=alm_formulation):
                self.assertEqual(
                    module.boozer_residual_threshold_for_stage(
                        stage,
                        constraint_method=constraint_method,
                        alm_formulation=alm_formulation,
                        alm_boozer_threshold=1.0e-4,
                    ),
                    expected,
                )

    def test_frontier_reference_metrics_share_boozer_objective_biot_savarts(self):
        module = self.load_module()
        averaged_objectives = []

        def average_surface_objectives(objectives):
            averaged_objectives.append(list(objectives))
            return SimpleNamespace(J=lambda: float(len(averaged_objectives)))

        surface_data = [
            {"boozer_surface": object()},
            {"boozer_surface": object()},
        ]

        with (
            self.patched_boozer_objective_term_types(module) as patched_types,
            patch.object(
                module,
                "average_surface_objectives",
                side_effect=average_surface_objectives,
            ),
        ):
            metrics = module.measure_frontier_reference_metrics(
                "final",
                surface_data,
                ["coil"],
            )

        self.assertEqual(metrics, (1.0, 2.0))
        self.assertEqual(len(patched_types.biot_savarts), 2)
        self.assertEqual(len(averaged_objectives), 2)
        self.assertIs(
            averaged_objectives[0][0].biotsavart,
            averaged_objectives[1][0].biotsavart,
        )
        self.assertIs(
            averaged_objectives[0][1].biotsavart,
            averaged_objectives[1][1].biotsavart,
        )
        self.assertIsInstance(
            averaged_objectives[1][0], patched_types.exact_residual_cls
        )

    def differentiable_residual_vector(
        self, surface, biotsavart, *, weight_inv_modB, I
    ):
        return biotsavart.residual_vector(
            surface,
            weight_inv_modB=weight_inv_modB,
            current_I=I,
        )

    def differentiable_residual_dB_derivatives(self, residual_size, nsurfdofs):
        residual_jacobian = np.zeros((residual_size, nsurfdofs + 2))
        second_by_field = np.zeros((residual_size, 3, nsurfdofs + 2))
        second_by_field_gradient = np.zeros((residual_size, 3, 3, nsurfdofs + 2))
        return residual_jacobian, second_by_field, second_by_field_gradient

    def differentiable_residual_dB(
        self,
        surface,
        iota,
        G,
        biotsavart,
        derivatives=0,
        weight_inv_modB=False,
        I=0.0,
        include_mixed_derivatives=True,
    ):
        del iota, G
        biotsavart.set_points(surface.gamma().reshape((-1, 3)))
        residual = self.differentiable_residual_vector(
            surface,
            biotsavart,
            weight_inv_modB=weight_inv_modB,
            I=I,
        )
        residual_dB = np.zeros((residual.size, 3))
        residual_dB[np.arange(residual.size), np.arange(residual.size) % 3] = 1.0
        if derivatives == 0:
            return residual, residual_dB
        nsurfdofs = surface.get_dofs().size
        derivative_terms = self.differentiable_residual_dB_derivatives(
            residual.size, nsurfdofs
        )
        if not include_mixed_derivatives:
            return residual, residual_dB, derivative_terms[0]
        return residual, residual_dB, *derivative_terms

    def tracked_differentiable_residual_dB(self, calls):
        def tracked_residual_dB(
            surface,
            iota,
            G,
            bs,
            derivatives=0,
            weight_inv_modB=False,
            I=0.0,
            include_mixed_derivatives=True,
        ):
            calls.append((derivatives, weight_inv_modB, I))
            return self.differentiable_residual_dB(
                surface,
                iota,
                G,
                bs,
                derivatives=derivatives,
                weight_inv_modB=weight_inv_modB,
                I=I,
                include_mixed_derivatives=include_mixed_derivatives,
            )

        return tracked_residual_dB

    @contextmanager
    def patched_boozer_residual_dB_evaluators(self, modules, residual_dB):
        # The residual_dB callable exists under two names depending on which
        # module is being patched: the upstream-only `boozer_surface_residual_dB`
        # in `simsopt.geo.surfaceobjectives`, and the I-aware
        # `boozer_surface_residual_dB_finite_I` re-exported through the
        # examples-side `banana_opt.boozer_residuals` and defined in
        # `banana_opt.boozer_finite_current`. Patch whichever symbol the
        # module exposes so existing call-tracking tests keep working without
        # caring about the wrapper rename.
        with ExitStack() as stack:
            for module in modules:
                stack.enter_context(
                    patch.object(module, "SurfaceXYZTensorFourier", FakeResolvedSurface)
                )
                if hasattr(module, "boozer_surface_residual_dB_finite_I"):
                    stack.enter_context(
                        patch.object(
                            module,
                            "boozer_surface_residual_dB_finite_I",
                            side_effect=residual_dB,
                        )
                    )
                if hasattr(module, "boozer_surface_residual_dB"):
                    stack.enter_context(
                        patch.object(
                            module,
                            "boozer_surface_residual_dB",
                            side_effect=residual_dB,
                        )
                    )
            yield

    def build_differentiable_boozer_case(
        self,
        *,
        current_I,
        weight_inv_modB,
        implicit_scale=(0.0, 0.0),
        res_type="exact",
    ):
        input_surface = FakeResolvedSurface(
            mpol=2,
            ntor=2,
            stellsym=True,
            nfp=5,
            quadpoints_phi=np.array([0.0, 0.07, 0.16]),
            quadpoints_theta=np.array([0.0, 0.31]),
        )
        input_surface.set_dofs(np.array([1.0, -2.0]))
        biotsavart = FakeDifferentiableBiotSavart(np.array([0.6, -0.35]))
        boozer_surface = FakeDifferentiableBoozerSurface(
            surface=input_surface,
            biotsavart=biotsavart,
            current_I=current_I,
            weight_inv_modB=weight_inv_modB,
            implicit_scale=implicit_scale,
            res_type=res_type,
        )
        return input_surface, boozer_surface, biotsavart

    def assert_directional_derivative_matches_fd(
        self,
        residual_cls,
        boozer_surface,
        biotsavart,
        *,
        eps=1.0e-6,
        **residual_kwargs,
    ):
        objective = residual_cls(boozer_surface, biotsavart, **residual_kwargs)
        direction = np.array([0.4, -0.7])
        x0 = biotsavart.x.copy()

        analytical = float(np.dot(objective.dJ(partials=True)(biotsavart), direction))

        biotsavart.x = x0 + eps * direction
        objective.recompute_bell()
        plus = objective.J()

        biotsavart.x = x0 - eps * direction
        objective.recompute_bell()
        minus = objective.J()

        biotsavart.x = x0
        objective.recompute_bell()

        finite_difference = (plus - minus) / (2.0 * eps)
        self.assertAlmostEqual(analytical, finite_difference, places=8)

    def run_exact_boozer_objective(self, module, *, current_I):
        input_surface = FakeResolvedSurface(
            mpol=2,
            ntor=2,
            stellsym=True,
            nfp=5,
            quadpoints_phi=np.linspace(0, 1 / 5, 2, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, 3, endpoint=False),
        )
        input_surface.set_dofs(np.array([1.0, -2.0]))
        fake_boozer_surface = FakeParentBoozerSurface(
            surface=input_surface,
            label=FakeLabel(),
            targetlabel=0.0,
            need_to_run_code=False,
            res={
                "iota": -0.4,
                "G": 1.2,
                "I": current_I,
                "type": "exact",
                "PLU": (None, None, None),
                "vjp": lambda adj, booz_surf, iota, G: np.zeros(2),
            },
        )
        fake_bs = FakeObjectiveBiotSavart()
        nsurfdofs = input_surface.get_dofs().size
        residual_dB_calls = []

        def fake_residual_dB(
            surface,
            iota,
            G,
            biotsavart,
            derivatives=0,
            weight_inv_modB=False,
            I=0.0,
            include_mixed_derivatives=True,
        ):
            num_points = 3 * surface.quadpoints_phi.size * surface.quadpoints_theta.size
            residual_dB_calls.append(
                (derivatives, weight_inv_modB, I, include_mixed_derivatives)
            )
            biotsavart.set_points(surface.gamma().reshape((-1, 3)))
            residual = np.ones(num_points)
            residual_dB = np.ones((num_points, 3))
            if derivatives == 0:
                return residual, residual_dB
            derivative_terms = self.differentiable_residual_dB_derivatives(
                num_points, nsurfdofs
            )
            if not include_mixed_derivatives:
                return residual, residual_dB, derivative_terms[0]
            return residual, residual_dB, *derivative_terms

        residual_module = self.residual_module(module)
        adjoint_solution = np.zeros(nsurfdofs + 2)
        with (
            self.patched_boozer_residual_dB_evaluators(
                (residual_module,),
                fake_residual_dB,
            ),
            patch.object(
                residual_module, "forward_backward", return_value=adjoint_solution
            ),
        ):
            objective = module.BoozerResidualExact(fake_boozer_surface, fake_bs)
            value = objective.J()
            gradient = objective.dJ(partials=True)

        return objective, fake_bs, residual_dB_calls, value, gradient

    def test_exact_boozer_residual_is_imported_from_banana_module(self):
        module = self.load_module()
        residual_module = self.residual_module(module)

        self.assertIs(module.BoozerResidualExact, residual_module.BoozerResidualExact)
        self.assertIs(
            module.RefinedBoozerResidual, residual_module.RefinedBoozerResidual
        )
        self.assertNotIn("boozer_surface_residual", vars(residual_module))
        # After the finite-I refactor, RefinedBoozerResidual evaluates the
        # residual through the examples-side wrapper rather than the upstream
        # vacuum function. Verify it pulls the wrapper symbol from
        # boozer_finite_current and re-exports it locally.
        from banana_opt.boozer_finite_current import boozer_surface_residual_dB_finite_I

        self.assertIs(
            residual_module.boozer_surface_residual_dB_finite_I,
            boozer_surface_residual_dB_finite_I,
        )
        self.assertNotIn("boozer_surface_residual_dB", vars(residual_module))
        self.assertIs(residual_module.forward_backward, forward_backward)

    def test_boozer_residual_exact_not_defined_inline(self):
        source = EXAMPLE_MODULE_PATH.read_text(encoding="utf-8")

        self.assertNotIn("class BoozerResidualExact", source)

    def test_save_surface_artifacts_writes_boozer_surface_jsons(self):
        module = self.load_module()

        class _Surface:
            def __init__(self):
                self.saved_paths = []
                self.vtk_paths = []

            def gamma(self):
                return np.zeros((1, 1, 3))

            def unitnormal(self):
                return np.ones((1, 1, 3))

            def to_vtk(self, path, extra_data=None):
                self.vtk_paths.append(path)

            def save(self, path):
                self.saved_paths.append(path)
                Path(path).write_text("surface", encoding="utf-8")

        class _BoozerSurface:
            def __init__(self, surface):
                self.surface = surface
                self.saved_paths = []

            def save(self, path):
                self.saved_paths.append(path)
                Path(path).write_text(
                    json.dumps(
                        {
                            "@module": "simsopt.geo.boozersurface",
                            "@class": "BoozerSurface",
                        }
                    ),
                    encoding="utf-8",
                )

        class _BiotSavart:
            def set_points(self, points):
                self.points = np.asarray(points)

            def B(self):
                return np.ones((1, 3))

        inner = _BoozerSurface(_Surface())
        outer = _BoozerSurface(_Surface())
        surface_data = [
            {"name": "inner", "boozer_surface": inner},
            {"name": "outer", "boozer_surface": outer},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            module.save_surface_artifacts(
                surface_data,
                _BiotSavart(),
                tmpdir,
                "surf_opt",
                also_write_outer_legacy=True,
            )

            tmp_path = Path(tmpdir)
            for filename in (
                "surf_opt_inner.json",
                "surf_opt_outer.json",
                "surf_opt.json",
                "surf_opt_inner_boozer_surface.json",
                "surf_opt_outer_boozer_surface.json",
                "surf_opt_boozer_surface.json",
            ):
                self.assertTrue((tmp_path / filename).exists())
            self.assertEqual(
                inner.saved_paths,
                [str(tmp_path / "surf_opt_inner_boozer_surface.json")],
            )
            self.assertEqual(
                outer.saved_paths,
                [
                    str(tmp_path / "surf_opt_outer_boozer_surface.json"),
                    str(tmp_path / "surf_opt_boozer_surface.json"),
                ],
            )

    def test_initialize_boozer_surface_exact_uses_ntor_phi_quadrature(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        boozer_surface = self.initialize_boozer_surface(
            module, surf_prev, constraint_weight=None
        )

        self.assertIsInstance(boozer_surface, FakeBoozerSurface)
        self.assertEqual(boozer_surface.targetlabel, TEST_VOL_TARGET)
        self.assertEqual(len(FakeSurfaceXYZTensorFourier.instances), 2)

        exact_surface = FakeSurfaceXYZTensorFourier.instances[1]
        expected_phi = np.linspace(
            0, 1 / surf_prev.nfp, 2 * TEST_NTOR + 1, endpoint=False
        )

        self.assertEqual(exact_surface.quadpoints_theta.size, 2 * TEST_MPOL + 1)
        self.assertEqual(exact_surface.quadpoints_phi.size, 2 * TEST_NTOR + 1)
        np.testing.assert_allclose(exact_surface.quadpoints_phi, expected_phi)

    def test_initialize_boozer_surface_zero_constraint_weight_keeps_least_squares_path(
        self,
    ):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        boozer_surface = self.initialize_boozer_surface(
            module, surf_prev, constraint_weight=0.0
        )

        self.assertIsInstance(boozer_surface, FakeBoozerSurface)
        self.assertEqual(boozer_surface.targetlabel, TEST_VOL_TARGET)
        self.assertEqual(len(FakeSurfaceXYZTensorFourier.instances), 1)
        self.assertIs(boozer_surface.surface, FakeSurfaceXYZTensorFourier.instances[0])
        self.assertEqual(boozer_surface.I, TEST_BOOZER_I)

    def test_initialize_boozer_surface_uses_loaded_surface_as_initial_guess(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        initial_surface_guess = SimpleNamespace(dofs=np.array([2.5, -1.5], dtype=float))

        boozer_surface = self.initialize_boozer_surface(
            module,
            surf_prev,
            constraint_weight=0.0,
            initial_surface_guess=initial_surface_guess,
        )

        self.assertIsInstance(boozer_surface, FakeBoozerSurface)
        self.assertEqual(len(FakeSurfaceXYZTensorFourier.instances), 1)
        seeded_surface = FakeSurfaceXYZTensorFourier.instances[0]
        np.testing.assert_allclose(seeded_surface.dofs, np.array([2.5, -1.5]))
        self.assertFalse(hasattr(seeded_surface, "fitted_gamma"))

    def test_initialize_boozer_surface_exact_threads_negative_current(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()

        with (
            patch.object(
                module, "SurfaceXYZTensorFourier", FakeSurfaceXYZTensorFourier
            ),
            patch.object(module, "Volume", FakeVolume),
        ):
            boozer_surface = module.initialize_boozer_surface(
                surf_prev,
                mpol=TEST_MPOL,
                ntor=TEST_NTOR,
                bs=object(),
                vol_target=TEST_VOL_TARGET,
                constraint_weight=None,
                iota=TEST_IOTA,
                G0=TEST_G0,
                boozer_I=-TEST_BOOZER_I,
                boozer_surface_cls=FakeBoozerSurface,
            )

        self.assertIsInstance(boozer_surface, FakeBoozerSurface)
        self.assertEqual(boozer_surface.I, -TEST_BOOZER_I)

    def test_real_boozersurface_source_treats_zero_constraint_weight_as_least_squares(
        self,
    ):
        source = BOOZER_SURFACE_PATH.read_text()
        self.assertIn(
            "self.boozer_type = 'ls' if constraint_weight is not None else 'exact'",
            source,
        )

    def test_boozer_residual_exact_threads_fixed_current_into_example_adjoint_path(
        self,
    ):
        module = self.load_module()
        objective, fake_bs, residual_dB_calls, value, gradient = (
            self.run_exact_boozer_objective(
                module,
                current_I=TEST_BOOZER_I,
            )
        )

        self.assertIsInstance(value, float)
        np.testing.assert_allclose(gradient, np.array([3.0, -1.0]))
        self.assertEqual(residual_dB_calls, [(1, True, TEST_BOOZER_I, False)])
        expected_point_count = (
            objective.surface.quadpoints_phi.size
            * objective.surface.quadpoints_theta.size
        )
        self.assertEqual(fake_bs.points.shape, (expected_point_count, 3))
        self.assertEqual(fake_bs.last_vjp_input.shape, (expected_point_count, 3))

    def test_refined_boozer_residual_k1_matches_standard_boozer_residual(self):
        # The vacuum-current branch is the only case where standard
        # BoozerResidual (now upstream/vacuum-only) equals RefinedBoozerResidual
        # at k=1. For finite I, RefinedBoozerResidual carries the iota*I*B
        # correction via the boozer_finite_current wrapper while standard
        # BoozerResidual does not. The finite-I divergence is asserted in the
        # wrapper-side tests in tests/geo/test_boozersurface.py.
        module = self.load_module()
        residual_module = self.residual_module(module)

        for current_I in (0.0,):
            for stored_weight_inv_modB in (False, True):
                with self.subTest(
                    current_I=current_I,
                    stored_weight_inv_modB=stored_weight_inv_modB,
                ):
                    input_surface, boozer_surface, biotsavart = (
                        self.build_differentiable_boozer_case(
                            current_I=current_I,
                            weight_inv_modB=stored_weight_inv_modB,
                            implicit_scale=(0.03, -0.02),
                        )
                    )
                    residual_calls = []
                    tracked_residual_dB = self.tracked_differentiable_residual_dB(
                        residual_calls
                    )

                    with self.patched_boozer_residual_dB_evaluators(
                        (surfaceobjectives_module, residual_module),
                        tracked_residual_dB,
                    ):
                        standard = surfaceobjectives_module.BoozerResidual(
                            boozer_surface, biotsavart
                        )
                        refined = residual_module.RefinedBoozerResidual(
                            boozer_surface,
                            biotsavart,
                            grid_multiplier=1,
                            include_label_constraint=True,
                            weight_inv_modB=None,
                        )

                        self.assertIs(standard.boozer_surface, refined.boozer_surface)
                        self.assertIs(standard.biotsavart, refined.biotsavart)
                        self.assertIs(
                            refined.surface.quadpoints_phi, input_surface.quadpoints_phi
                        )
                        self.assertIs(
                            refined.surface.quadpoints_theta,
                            input_surface.quadpoints_theta,
                        )
                        self.assertEqual(standard.J(), refined.J())
                        np.testing.assert_allclose(
                            standard.dJ(partials=True)(biotsavart),
                            refined.dJ(partials=True)(biotsavart),
                            rtol=1.0e-13,
                            atol=1.0e-13,
                        )

                    self.assertEqual(
                        {call[1] for call in residual_calls}, {stored_weight_inv_modB}
                    )
                    self.assertEqual({call[2] for call in residual_calls}, {current_I})

    def test_refined_boozer_residual_ls_uses_cached_adjoint_state(self):
        module = self.load_module()
        residual_module = self.residual_module(module)
        _, boozer_surface, biotsavart = self.build_differentiable_boozer_case(
            current_I=TEST_BOOZER_I,
            weight_inv_modB=True,
            res_type="ls",
        )
        residual_calls = []
        tracked_residual_dB = self.tracked_differentiable_residual_dB(residual_calls)

        with self.patched_boozer_residual_dB_evaluators(
            (residual_module,),
            tracked_residual_dB,
        ):
            objective = residual_module.RefinedBoozerResidual(
                boozer_surface,
                biotsavart,
                grid_multiplier=1,
                include_label_constraint=True,
                weight_inv_modB=None,
            )
            objective.J()
            objective.dJ(partials=True)(biotsavart)

        self.assertEqual(residual_calls, [(1, True, TEST_BOOZER_I)])
        self.assertEqual(biotsavart.b_and_dB_vjp_calls, 1)

    def test_refined_boozer_residual_threshold_clamps_value_and_gradient(self):
        module = self.load_module()
        residual_module = self.residual_module(module)
        _, boozer_surface, biotsavart = self.build_differentiable_boozer_case(
            current_I=TEST_BOOZER_I,
            weight_inv_modB=True,
        )

        with self.patched_boozer_residual_dB_evaluators(
            (residual_module,),
            self.differentiable_residual_dB,
        ):
            raw = residual_module.RefinedBoozerResidual(
                boozer_surface,
                biotsavart,
                grid_multiplier=1,
                include_label_constraint=True,
                weight_inv_modB=None,
            )
            raw_value = raw.J()
            raw_gradient = raw.dJ(partials=True)(biotsavart)
            active = residual_module.RefinedBoozerResidual(
                boozer_surface,
                biotsavart,
                grid_multiplier=1,
                include_label_constraint=True,
                weight_inv_modB=None,
                threshold=raw_value / 2.0,
            )
            inactive = residual_module.RefinedBoozerResidual(
                boozer_surface,
                biotsavart,
                grid_multiplier=1,
                include_label_constraint=True,
                weight_inv_modB=None,
                threshold=raw_value * 2.0,
            )

            self.assertGreater(raw_value, 0.0)
            self.assertAlmostEqual(active.J(), raw_value / 2.0)
            np.testing.assert_allclose(
                active.dJ(partials=True)(biotsavart),
                raw_gradient,
            )
            self.assertEqual(inactive.J(), 0.0)
            np.testing.assert_allclose(
                inactive.dJ(partials=True)(biotsavart),
                np.zeros_like(raw_gradient),
            )
            np.testing.assert_allclose(
                inactive.dJ_by_dB(),
                np.zeros_like(raw.dJ_by_dB()),
            )

    def test_boozer_residual_exact_matches_refined_k4_compatibility_config(self):
        module = self.load_module()
        residual_module = self.residual_module(module)
        input_surface, boozer_surface, biotsavart = (
            self.build_differentiable_boozer_case(
                current_I=TEST_BOOZER_I,
                weight_inv_modB=False,
            )
        )

        with self.patched_boozer_residual_dB_evaluators(
            (residual_module,),
            self.differentiable_residual_dB,
        ):
            exact = residual_module.BoozerResidualExact(boozer_surface, biotsavart)
            refined = residual_module.RefinedBoozerResidual(
                boozer_surface,
                biotsavart,
                grid_multiplier=4,
                include_label_constraint=True,
                weight_inv_modB=True,
            )

            expected_phi = np.linspace(
                0,
                1.0 / input_surface.nfp,
                input_surface.quadpoints_phi.size * 4,
                endpoint=False,
            )
            expected_theta = np.linspace(
                0,
                1,
                input_surface.quadpoints_theta.size * 4,
                endpoint=False,
            )
            np.testing.assert_allclose(exact.surface.quadpoints_phi, expected_phi)
            np.testing.assert_allclose(exact.surface.quadpoints_theta, expected_theta)
            np.testing.assert_allclose(refined.surface.quadpoints_phi, expected_phi)
            np.testing.assert_allclose(refined.surface.quadpoints_theta, expected_theta)
            self.assertTrue(exact.include_label_constraint)
            self.assertEqual(exact.J(), refined.J())
            np.testing.assert_allclose(
                exact.dJ(partials=True)(biotsavart),
                refined.dJ(partials=True)(biotsavart),
                rtol=1.0e-13,
                atol=1.0e-13,
            )

    def test_refined_boozer_residual_rejects_label_constraint_without_weight(self):
        module = self.load_module()
        residual_module = self.residual_module(module)
        _, boozer_surface, biotsavart = self.build_differentiable_boozer_case(
            current_I=TEST_BOOZER_I,
            weight_inv_modB=True,
        )
        boozer_surface.constraint_weight = None

        with self.assertRaisesRegex(ValueError, "numeric constraint_weight"):
            residual_module.RefinedBoozerResidual(
                boozer_surface,
                biotsavart,
                include_label_constraint=True,
            )

    def test_boozer_terms_omit_label_penalty_for_exact_constrained_surface(self):
        module = self.load_module()
        residual_module = self.residual_module(module)
        _, boozer_surface, biotsavart = self.build_differentiable_boozer_case(
            current_I=TEST_BOOZER_I,
            weight_inv_modB=True,
        )
        boozer_surface.constraint_weight = None

        with (
            self.patched_boozer_residual_dB_evaluators(
                (residual_module,),
                self.differentiable_residual_dB,
            ),
            patch.object(module, "BiotSavart", return_value=biotsavart),
            patch.object(
                module,
                "NonQuasiSymmetricRatio",
                return_value=FakeAlgebraicObjective(0.0, [0.0, 0.0]),
            ),
        ):
            terms = module.build_boozer_derived_objective_terms(
                "initial",
                [{"boozer_surface": boozer_surface}],
                [],
            )
            residual = terms["brs"][0]
            self.assertFalse(residual.include_label_constraint)
            self.assertIsInstance(residual.J(), float)

    def test_boozer_residual_exact_omits_label_penalty_for_exact_constrained_surface(
        self,
    ):
        module = self.load_module()
        residual_module = self.residual_module(module)
        _, boozer_surface, biotsavart = self.build_differentiable_boozer_case(
            current_I=TEST_BOOZER_I,
            weight_inv_modB=True,
        )
        boozer_surface.constraint_weight = None

        with self.patched_boozer_residual_dB_evaluators(
            (residual_module,),
            self.differentiable_residual_dB,
        ):
            residual = residual_module.BoozerResidualExact(boozer_surface, biotsavart)
            self.assertFalse(residual.include_label_constraint)
            self.assertIsInstance(residual.J(), float)

    def test_refined_boozer_residual_rejects_invalid_grid_multiplier(self):
        module = self.load_module()
        residual_module = self.residual_module(module)
        _, boozer_surface, biotsavart = self.build_differentiable_boozer_case(
            current_I=0.0,
            weight_inv_modB=True,
        )

        with self.assertRaises(ValueError):
            residual_module.RefinedBoozerResidual(
                boozer_surface,
                biotsavart,
                grid_multiplier=0,
            )

    def test_refined_boozer_residual_explicit_false_weight_override_wins(self):
        module = self.load_module()
        residual_module = self.residual_module(module)
        _, boozer_surface, biotsavart = self.build_differentiable_boozer_case(
            current_I=TEST_BOOZER_I,
            weight_inv_modB=True,
        )
        residual_calls = []
        tracked_residual_dB = self.tracked_differentiable_residual_dB(residual_calls)

        with self.patched_boozer_residual_dB_evaluators(
            (residual_module,),
            tracked_residual_dB,
        ):
            objective = residual_module.RefinedBoozerResidual(
                boozer_surface,
                biotsavart,
                grid_multiplier=4,
                include_label_constraint=False,
                weight_inv_modB=False,
            )
            objective.J()

        self.assertEqual({call[1] for call in residual_calls}, {False})

    def test_refined_boozer_residual_directional_derivatives(self):
        module = self.load_module()
        residual_module = self.residual_module(module)

        cases = (
            (1, True, None),
            (4, False, True),
            (4, True, None),
        )
        for grid_multiplier, include_label_constraint, weight_inv_modB in cases:
            with self.subTest(
                grid_multiplier=grid_multiplier,
                include_label_constraint=include_label_constraint,
                weight_inv_modB=weight_inv_modB,
            ):
                _, boozer_surface, biotsavart = self.build_differentiable_boozer_case(
                    current_I=TEST_BOOZER_I,
                    weight_inv_modB=False,
                )
                with self.patched_boozer_residual_dB_evaluators(
                    (residual_module,),
                    self.differentiable_residual_dB,
                ):
                    self.assert_directional_derivative_matches_fd(
                        residual_module.RefinedBoozerResidual,
                        boozer_surface,
                        biotsavart,
                        grid_multiplier=grid_multiplier,
                        include_label_constraint=include_label_constraint,
                        weight_inv_modB=weight_inv_modB,
                    )

    def test_boozer_residual_exact_no_longer_accepts_unused_constraint_weight(self):
        module = self.load_module()

        with self.assertRaises(TypeError):
            module.BoozerResidualExact(object(), object(), constraint_weight=0.0)

    def test_resolve_plasma_current_settings_accepts_physical_amps(self):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=None,
                plasma_current_A=8000.0,
            )
        )

        self.assertEqual(settings["input_source"], "physical_A")
        self.assertEqual(settings["mode"], "wataru_proxy_field")
        self.assertEqual(settings["effective_mode"], "wataru_proxy_field")
        self.assertEqual(settings["plasma_current_A"], 8000.0)
        self.assertAlmostEqual(settings["boozer_I"], 4.0e-7 * np.pi * 8000.0)
        self.assertEqual(settings["boozer_current_convention"], "mu0")

    def test_resolve_plasma_current_settings_zero_physical_amps_reports_vacuum_effective_mode(
        self,
    ):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=None,
                plasma_current_A=0.0,
            )
        )

        self.assertEqual(settings["input_source"], "physical_A")
        self.assertEqual(settings["mode"], "wataru_proxy_field")
        self.assertEqual(settings["effective_mode"], "vacuum")
        self.assertEqual(settings["plasma_current_A"], 0.0)
        self.assertEqual(settings["boozer_I"], 0.0)

    def test_resolve_plasma_current_settings_accepts_negative_physical_amps(self):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=None,
                plasma_current_A=-35200.0,
            )
        )

        self.assertEqual(settings["plasma_current_A"], -35200.0)
        self.assertAlmostEqual(settings["boozer_I"], 4.0e-7 * np.pi * -35200.0)
        self.assertEqual(settings["effective_mode"], "wataru_proxy_field")

    def test_resolve_plasma_current_settings_rejects_mixed_raw_and_physical_inputs(
        self,
    ):
        module = self.load_module()

        with self.assertRaisesRegex(ValueError, "--plasma-current-A"):
            module.resolve_plasma_current_settings(
                SimpleNamespace(
                    boozer_I=0.5,
                    plasma_current_A=8000.0,
                )
            )

    def test_resolve_plasma_current_settings_defaults_to_surrogate_zero(self):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=None,
                plasma_current_A=None,
            )
        )

        self.assertEqual(settings["input_source"], "default_zero")
        self.assertEqual(settings["mode"], "wataru_proxy_field")
        self.assertEqual(settings["effective_mode"], "vacuum")
        self.assertEqual(settings["plasma_current_A"], 0.0)
        self.assertEqual(settings["boozer_I"], 0.0)
        self.assertEqual(settings["boozer_current_convention"], "mu0")

    def test_resolve_plasma_current_settings_single_surface_normalizes_legacy_mode(
        self,
    ):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=None,
                plasma_current_A=9100.0,
                finite_current_mode=None,
            ),
            finite_current_mode="boozer_surrogate",
            num_surfaces=1,
        )

        self.assertEqual(settings["mode"], "wataru_proxy_field")
        self.assertEqual(settings["effective_mode"], "wataru_proxy_field")
        self.assertEqual(settings["input_source"], "physical_A")
        self.assertAlmostEqual(settings["boozer_I"], 4.0e-7 * np.pi * 9100.0)

    def test_resolve_plasma_current_settings_single_surface_allows_raw_boozer_override(
        self,
    ):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=0.125,
                plasma_current_A=None,
                finite_current_mode=None,
            ),
            finite_current_mode="boozer_surrogate",
            num_surfaces=1,
        )

        self.assertEqual(settings["mode"], "wataru_proxy_field")
        self.assertEqual(settings["input_source"], "raw_boozer_I")
        self.assertAlmostEqual(settings["boozer_I"], 0.125)
        self.assertAlmostEqual(settings["plasma_current_A"], 0.125 / (4.0e-7 * np.pi))

    def test_resolve_plasma_current_settings_single_surface_allows_jhalpern_mode(self):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=None,
                plasma_current_A=None,
                finite_current_mode="jhalpern30_proxy_field",
            ),
            finite_current_mode="jhalpern30_proxy_field",
            default_plasma_current_A=-6500.0,
            num_surfaces=1,
        )

        self.assertEqual(settings["mode"], "jhalpern30_proxy_field")
        self.assertEqual(settings["effective_mode"], "jhalpern30_proxy_field")
        self.assertEqual(settings["input_source"], "artifact_default_A")
        self.assertAlmostEqual(settings["boozer_I"], 4.0e-7 * np.pi * -6500.0)

    def test_resolve_plasma_current_settings_single_surface_rejects_conflicting_requested_mode(
        self,
    ):
        module = self.load_module()

        with self.assertRaisesRegex(ValueError, "Single-surface mode is locked to"):
            module.resolve_plasma_current_settings(
                SimpleNamespace(
                    boozer_I=None,
                    plasma_current_A=9100.0,
                    finite_current_mode="boozer_surrogate",
                ),
                finite_current_mode="boozer_surrogate",
                num_surfaces=1,
            )

    def test_resolve_plasma_current_settings_multisurface_preserves_requested_mode(
        self,
    ):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=None,
                plasma_current_A=6400.0,
                finite_current_mode="boozer_surrogate",
            ),
            finite_current_mode="boozer_surrogate",
            num_surfaces=2,
        )

        self.assertEqual(settings["mode"], "boozer_surrogate")
        self.assertEqual(settings["effective_mode"], "boozer_surrogate")
        self.assertAlmostEqual(settings["boozer_I"], 4.0e-7 * np.pi * 6400.0)

    def test_resolve_plasma_current_settings_uses_artifact_default_in_wataru_mode(self):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=None,
                plasma_current_A=None,
            ),
            finite_current_mode="wataru_proxy_field",
            default_plasma_current_A=8000.0,
        )

        self.assertEqual(settings["input_source"], "artifact_default_A")
        self.assertEqual(settings["mode"], "wataru_proxy_field")
        self.assertEqual(settings["effective_mode"], "wataru_proxy_field")
        self.assertEqual(settings["plasma_current_A"], 8000.0)
        self.assertAlmostEqual(settings["boozer_I"], 4.0e-7 * np.pi * 8000.0)
        self.assertEqual(settings["boozer_current_convention"], "mu0")

    def test_wataru_mode_preserves_mu0_no_2pi_convention(self):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=None,
                plasma_current_A=9000.0,
            ),
            finite_current_mode="wataru_proxy_field",
        )

        self.assertEqual(settings["boozer_current_convention"], "mu0")
        self.assertAlmostEqual(settings["boozer_I"], 4.0e-7 * np.pi * 9000.0)
        self.assertNotAlmostEqual(settings["boozer_I"], 2.0e-7 * 9000.0)

    def test_stage2_resolve_finite_current_mode_accepts_explicit_wataru_without_artifact(
        self,
    ):
        module = load_stage2_module()

        self.assertEqual(
            module.resolve_finite_current_mode(
                "wataru_proxy_field",
                artifact_mode=None,
            ),
            "wataru_proxy_field",
        )

    def test_stage2_resolve_finite_current_mode_explains_legacy_assumed_default(self):
        module = load_stage2_module()

        with self.assertRaisesRegex(
            ValueError,
            "recorded no finite-current mode, so that value was assumed as the legacy default",
        ):
            module.resolve_finite_current_mode(
                "wataru_proxy_field",
                artifact_mode="boozer_surrogate",
                artifact_mode_source="legacy_assumed_default",
            )

    def test_build_stage2_bs_path_uses_unique_globbed_current_match(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parent = root / "local" / "outputs-demo.nc"
            parent.mkdir(parents=True)
            matched = (
                parent
                / "R0=0.976-s=0.24-LW=0.0005-CCW=100-CCT=0.05-CW=0.0001-CT=40-SR=0.220-INITC=-10000-MAXC=16000-TFC=-80000-Order=2-CM=penalty-BH=3"
                / "biot_savart_opt.json"
            )
            matched.parent.mkdir(parents=True)
            matched.write_text("{}", encoding="utf-8")

            args = SimpleNamespace(
                stage2_bs_path=None,
                plasma_surf_filename="demo.nc",
                stage2_seed_major_radius=0.976,
                stage2_seed_toroidal_flux=0.24,
                stage2_seed_length_weight=0.0005,
                stage2_seed_cc_weight=100.0,
                stage2_seed_cc_threshold=0.05,
                stage2_seed_curvature_weight=0.0001,
                stage2_seed_curvature_threshold=40.0,
                stage2_seed_banana_surf_radius=0.22,
                stage2_seed_tf_current_A=-8.0e4,
                stage2_seed_order=2,
                stage2_seed_banana_init_current_A=-1.0e4,
                stage2_source="local",
                local_stage2_root=str(root / "local"),
                database_stage2_root=str(root / "database"),
            )

            self.assertEqual(module.build_stage2_bs_path(args), str(matched))

    def test_build_stage2_bs_path_rejects_ambiguous_globbed_current_matches(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parent = root / "local" / "outputs-demo.nc"
            parent.mkdir(parents=True)
            for suffix in ("-CM=penalty-BH=3", "-CM=penalty-BH=4"):
                candidate = (
                    parent
                    / (
                        "R0=0.976-s=0.24-LW=0.0005-CCW=100-CCT=0.05-CW=0.0001-CT=40-"
                        f"SR=0.220-INITC=-10000-MAXC=16000-TFC=-80000-Order=2{suffix}"
                    )
                    / "biot_savart_opt.json"
                )
                candidate.parent.mkdir(parents=True)
                candidate.write_text("{}", encoding="utf-8")

            args = SimpleNamespace(
                stage2_bs_path=None,
                plasma_surf_filename="demo.nc",
                stage2_seed_major_radius=0.976,
                stage2_seed_toroidal_flux=0.24,
                stage2_seed_length_weight=0.0005,
                stage2_seed_cc_weight=100.0,
                stage2_seed_cc_threshold=0.05,
                stage2_seed_curvature_weight=0.0001,
                stage2_seed_curvature_threshold=40.0,
                stage2_seed_banana_surf_radius=0.22,
                stage2_seed_tf_current_A=-8.0e4,
                stage2_seed_order=2,
                stage2_seed_banana_init_current_A=-1.0e4,
                stage2_source="local",
                local_stage2_root=str(root / "local"),
                database_stage2_root=str(root / "database"),
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "Multiple Stage 2 outputs match the requested seed specification",
            ):
                module.build_stage2_bs_path(args)

    def test_fun_fallback_returns_elevated_j_and_same_sign_gradient(self):
        """Issue #2: failed Boozer must return elevated J + same-sign gradient,
        not (J_old, -dJ_old)."""
        module = self.load_module()

        last_J = 42.0
        last_dJ = np.array([1.0, -2.0, 3.0, -4.0, 5.0])

        class _Surface:
            x = np.ones(3)

            def is_self_intersecting(self):
                return False

            def volume(self):
                return 1.0

            def gamma(self):
                return np.zeros((1, 1, 3))

        class _BoozerSurface:
            surface = _Surface()
            res = {"success": False, "iota": TEST_IOTA, "G": TEST_G0}

            def run_code(self, iota, G):
                return self.res

        class _JF:
            x = np.zeros(5)

        surface_data = [{"boozer_surface": _BoozerSurface()}]
        module.run_dict = {
            "x_prev": np.zeros(5),
            "lscount": 0,
            "surface_state": {
                "sdofs": [np.ones(3)],
                "iota": [TEST_IOTA],
                "G": [TEST_G0],
            },
            "J": last_J,
            "dJ": last_dJ.copy(),
            "accepted_iterations": 0,
            "accepted_x": np.zeros(5),
        }
        module.surface_data = surface_data
        module.VV = object()
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.JF = _JF()

        J_out, dJ_out = module.fun(np.ones(5))

        self.assertGreater(J_out, last_J)
        np.testing.assert_array_equal(dJ_out, last_dJ)
        self.assertIsNot(dJ_out, module.run_dict["dJ"])

    def test_evaluate_topology_gate_reports_early_surface_exit(self):
        module = self.load_module()
        status = module._evaluate_topology_gate_impl(
            object(),
            object(),
            2,
            2.0,
            1e-7,
            0.75,
            score_topology_fn=lambda *_args, **_kwargs: _mock_topology_score_result(
                stop_reason="surface_exit",
                first_exit_time=0.8,
            ),
        )

        self.assertTrue(status["enabled"])
        self.assertFalse(status["success"])
        self.assertEqual(status["state"], "modeled_infeasible")
        self.assertEqual(status["survived_lines"], 1)
        self.assertAlmostEqual(status["survival_fraction"], 0.5)
        self.assertEqual(status["first_exit_reason"], "surface_exit")
        self.assertAlmostEqual(status["first_exit_time"], 0.8)

    def test_evaluate_topology_gate_threads_transport_diagnostics(self):
        module = self.load_module()
        transport_diagnostics = {
            "schema_version": "single_stage_topology_transport_diagnostics_v1",
            "status": "partial",
            "gamma_c": {"status": "unavailable"},
            "effective_ripple": {"status": "unavailable", "aliases": ["epsilon_eff"]},
        }

        status = module._evaluate_topology_gate_impl(
            object(),
            object(),
            2,
            2.0,
            1e-7,
            0.75,
            score_topology_fn=lambda *_args, **_kwargs: {
                **_mock_topology_score_result(
                    stop_reason="surface_exit",
                    first_exit_time=0.8,
                ),
                "transport_diagnostics": transport_diagnostics,
            },
        )

        self.assertEqual(status["transport_diagnostics"], transport_diagnostics)

    def test_evaluate_topology_gate_wrapper_uses_shared_impl_signature(self):
        module = self.load_module()

        with patch.object(
            module,
            "_evaluate_topology_gate_impl",
            return_value={"ok": True},
        ) as gate_impl:
            result = module.evaluate_topology_gate(
                "surface",
                "field",
                2,
                2.0,
                1e-7,
                0.75,
            )

        self.assertEqual(result, {"ok": True})
        gate_impl.assert_called_once_with(
            "surface",
            "field",
            2,
            2.0,
            1e-7,
            0.75,
            topology_seed_mode=module.SEED_MODE_MIDPLANE,
        )

    def test_evaluate_topology_gate_marks_iteration_limit_as_broken(self):
        module = self.load_module()
        status = module._evaluate_topology_gate_impl(
            object(),
            object(),
            2,
            2.0,
            1e-7,
            0.5,
            score_topology_fn=lambda *_args, **_kwargs: _mock_topology_score_result(
                stop_reason="iteration_limit",
                first_exit_time=0.4,
            ),
        )

        self.assertFalse(status["success"])
        self.assertEqual(status["state"], "broken")
        self.assertTrue(status["broken"])
        self.assertEqual(status["survived_lines"], 1)
        self.assertAlmostEqual(status["survival_fraction"], 0.5)

    def test_disabled_topology_gate_status_does_not_claim_feasible_state(self):
        module = self.load_module()

        status = module.disabled_topology_gate_status(2.0, 1e-7, 0.25)

        self.assertFalse(status["enabled"])
        self.assertTrue(status["success"])
        self.assertIsNone(status["state"])
        self.assertFalse(status["broken"])
        diagnostics = module.build_topology_gate_diagnostics(
            status,
            artifact_role="final_topology_gate",
        )
        self.assertEqual(diagnostics["outcome"], "disabled")

    def test_topology_gate_and_scorer_share_trace_metrics(self):
        module = self.load_module()
        topology_module = load_topology_scorer_module()

        class _Surface:
            nfp = 1

            def cross_section(self, *, phi, thetas):
                theta = np.linspace(0.0, 2.0 * np.pi, int(thetas), endpoint=False)
                return np.column_stack(
                    (
                        1.1 + 0.1 * np.cos(theta),
                        np.zeros_like(theta) + float(phi),
                        0.1 * np.sin(theta),
                    )
                )

        fieldlines_tys = [
            np.array([[0.0, 1.0, 0.0, 0.0]]),
            np.array([[0.0, 1.1, 0.0, 0.0]]),
            np.array([[0.0, 1.2, 0.0, 0.0]]),
        ]
        fieldlines_phi_hits = [
            np.array([[0.4, 0.0, 1.0, 0.0, 0.0], [0.7, -1.0, 1.0, 0.0, 0.0]]),
            np.array([[0.5, 0.0, 1.1, 0.0, 0.0]]),
            np.array([]),
        ]
        stop_labels = [
            "surface_exit",
            "max_z_guardrail",
            "min_z_guardrail",
            "min_r_guardrail",
            "max_r_guardrail",
            "iteration_limit",
        ]

        with (
            patch.object(
                topology_module,
                "build_stopping_criteria",
                return_value=([object()], stop_labels),
            ),
            patch.object(
                topology_module,
                "midplane_seed_radii",
                return_value=np.array([1.0, 1.1, 1.2]),
            ),
            patch.object(
                topology_module,
                "prepare_topology_field",
                return_value=(object(), {"selected_mode": "native"}),
            ),
            patch.object(
                topology_module,
                "cross_section_span",
                return_value=1.0,
            ),
            patch.object(
                topology_module,
                "invariant_torus_classification",
                return_value={
                    "invariant_torus_fraction": None,
                    "invariant_torus_count": 0,
                    "wba_fraction_denominator_policy": None,
                    "wba_fraction_denominator_seed_count": 0,
                    "wba_min_classifiable_seeds": 2,
                    "wba_not_evaluated_seed_count": 0,
                    "wba_seed_count": 0,
                    "wba_survived_seed_count": 0,
                    "wba_classified_seed_count": 0,
                    "wba_evaluation_state": "not_evaluated_no_classified_seeds",
                    "wba_not_evaluated_reason": "not_evaluated_no_classified_seeds",
                    "wba_classification_counts": {},
                    "wba_rotation_number_median": None,
                    "wba_matching_digits_min": None,
                    "wba_matching_digits_median": None,
                    "wba_seed_classifications": [],
                    "wba_axis": None,
                    "wba_poincare_plane_index": 0,
                    "wba_settings": {},
                },
            ),
            patch(
                "simsopt.field.compute_fieldlines",
                return_value=(fieldlines_tys, fieldlines_phi_hits),
            ),
        ):
            scorer_result = topology_module.score_topology(
                _Surface(),
                object(),
                nfieldlines=3,
                tmax=2.0,
                tol=1e-7,
                nphis=1,
                field_policy="never",
            )
        gate_status = module._evaluate_topology_gate_impl(
            _Surface(),
            object(),
            3,
            2.0,
            1e-7,
            0.60,
            score_topology_fn=lambda *_args, **_kwargs: scorer_result,
        )

        self.assertAlmostEqual(
            gate_status["survival_fraction"],
            scorer_result["survival_fraction"],
        )
        self.assertEqual(
            gate_status["survived_lines"],
            scorer_result["survived_lines"],
        )
        self.assertEqual(
            gate_status["stop_reason_counts"],
            scorer_result["stop_reason_counts"],
        )
        self.assertAlmostEqual(
            gate_status["first_exit_time"],
            scorer_result["first_exit"]["first_exit_time"],
        )
        self.assertAlmostEqual(
            gate_status["first_exit_angle"],
            scorer_result["first_exit"]["first_exit_angle"],
        )

    def test_topology_gate_skips_wba_axis_classification(self):
        module = self.load_module()
        captured_kwargs = {}

        def fake_score_topology(_surface, _bfield, **kwargs):
            captured_kwargs.update(kwargs)
            return {
                "survival_fraction": 1.0,
                "survived_lines": 2,
                "stop_reason_counts": {"complete": 2},
                "first_exit": None,
                "seed_contract": {"mode": "midplane_radial_sweep"},
                "field_model": {"selected_mode": "native"},
                "transport_diagnostics": {"evaluation_state": "not_evaluated"},
            }

        gate_status = module._evaluate_topology_gate_impl(
            object(),
            object(),
            2,
            2.0,
            1e-7,
            0.60,
            score_topology_fn=fake_score_topology,
        )

        self.assertTrue(gate_status["success"])
        self.assertFalse(captured_kwargs["compute_transport_diagnostics"])
        self.assertFalse(captured_kwargs["compute_invariant_torus_classification"])

    def test_topology_scorer_safe_wrapper_returns_broken_result_on_exception(self):
        module = load_topology_scorer_module()

        with patch.object(
            module,
            "score_topology",
            side_effect=RuntimeError("trace exploded"),
        ):
            result = module.safe_score_topology(
                object(),
                object(),
                nfieldlines=4,
                tmax=2.0,
            )

        self.assertTrue(result["broken"])
        self.assertEqual(result["evaluation_state"], "broken")
        self.assertIn("trace exploded", result["evaluation_error"])
        self.assertEqual(result["evaluation_error_type"], "RuntimeError")
        self.assertEqual(result["nfieldlines"], 4)
        self.assertEqual(result["survived_lines"], 0)
        self.assertTrue(np.isinf(result["confinement_loss"]))

    def test_legacy_bounded_seed_fraction_counts_empty_hit_rows_as_unbounded_seeds(
        self,
    ):
        module = load_topology_scorer_module()
        bounded_line = np.array(
            [
                [0.0, 0.0, 1.00, 0.0, 0.00],
                [1.0, 0.0, 1.01, 0.0, 0.01],
                [2.0, 0.0, 1.02, 0.0, 0.00],
            ],
            dtype=float,
        )

        fraction, median_width = module.legacy_bounded_seed_fraction(
            [np.empty((0, 5)), bounded_line],
            cross_section_span=1.0,
            width_ratio=0.25,
        )

        self.assertAlmostEqual(fraction, 0.5)
        self.assertAlmostEqual(median_width, 0.5 * (1.0 + np.sqrt(0.02**2 + 0.01**2)))

    def test_trace_metrics_rejects_malformed_empty_hit_rows(self):
        topology_module = load_topology_scorer_module()

        with self.assertRaises(ValueError):
            topology_module.trace_metrics(
                [np.array([[0.0, 1.0, 0.0, 0.0]])],
                [np.empty((0, 0))],
                [],
                ["surface_exit"],
            )

    def test_trace_metrics_marks_iteration_limit_as_broken_validation(self):
        topology_module = load_topology_scorer_module()

        metrics = topology_module.trace_metrics(
            [np.array([[0.0, 1.0, 0.0, 0.0]])],
            [np.array([[0.1, -6.0, 1.0, 0.0, 0.0]])],
            [0.0],
            [
                "surface_exit",
                "max_z_guardrail",
                "min_z_guardrail",
                "min_r_guardrail",
                "max_r_guardrail",
                "iteration_limit",
            ],
            mode="validation",
        )

        self.assertEqual(metrics["validation_status"], "broken")
        self.assertEqual(metrics["stop_reason_counts"]["iteration_limit"], 1)

    def test_trace_metrics_refuses_clean_verdict_on_underresolved_high_iota_surface(self):
        topology_module = load_topology_scorer_module()
        # One field line that survives with no stop -> would be a clean "validated" 50/50.
        survived_tys = [np.array([[0.0, 1.0, 0.0, 0.0]])]
        survived_hits = [np.array([[1.0, 0.0, 1.0, 0.0, 0.0]])]
        phis = [0.0]
        labels = ["surface_exit"]

        def status(**kw):
            return topology_module.trace_metrics(
                survived_tys, survived_hits, phis, labels, mode="validation", **kw
            )["validation_status"]

        # Backward-compatible: without iota/resolution the verdict is unchanged.
        self.assertEqual(status(), "validated")
        # High iota on a coarse 127/32 surface: refuse the clean verdict (the footgun).
        self.assertEqual(
            status(iota=0.30, surface_resolution=(127, 32)), "under_resolved"
        )
        # ntheta alone under the floor still trips the guard.
        self.assertEqual(
            status(iota=0.30, surface_resolution=(255, 32)), "under_resolved"
        )
        # High iota on the fine 255/64 surface: trusted -> validated.
        self.assertEqual(status(iota=0.30, surface_resolution=(255, 64)), "validated")
        # Low iota is unaffected by the guard even on a coarse surface.
        self.assertEqual(status(iota=0.10, surface_resolution=(127, 32)), "validated")

    def test_legacy_bounded_seed_fraction_and_trace_metrics_share_first_stop_semantics(
        self,
    ):
        topology_module = load_topology_scorer_module()
        hits = np.array(
            [
                [0.0, 0.0, 1.00, 0.0, 0.00],
                [1.0, 0.0, 1.01, 0.0, 0.01],
                [2.0, -1.0, 1.01, 0.0, 0.01],
                [3.0, 0.0, 1.02, 0.0, 0.00],
                [4.0, 0.0, 1.03, 0.0, 0.01],
            ],
            dtype=float,
        )

        metrics = topology_module.trace_metrics(
            [np.array([[0.0, 1.0, 0.0, 0.0]])],
            [hits],
            [0.0],
            ["surface_exit"],
            mode="validation",
        )
        fraction, median_width = topology_module.legacy_bounded_seed_fraction(
            [hits],
            cross_section_span=1.0,
            width_ratio=0.25,
        )

        self.assertEqual(metrics["survived_lines"], 0)
        self.assertEqual(metrics["per_phi_hit_counts"], [2])
        self.assertEqual(fraction, 0.0)
        self.assertEqual(median_width, 1.0)

    def test_midplane_seed_radii_produces_inset_radial_sweep(self):
        topology_module = load_topology_scorer_module()

        class _Surface:
            nfp = 5

            def cross_section(self, phi, thetas):
                angles = np.linspace(0.0, 2.0 * np.pi, int(thetas), endpoint=False)
                R = 1.0 + 0.2 * np.cos(angles)
                Z = 0.15 * np.sin(angles)
                phi_abs = 2.0 * np.pi * float(phi)
                return np.column_stack(
                    [
                        R * np.cos(phi_abs),
                        R * np.sin(phi_abs),
                        Z,
                    ]
                )

        radii = topology_module.midplane_seed_radii(_Surface(), 12, inset_fraction=0.05)
        self.assertEqual(radii.shape, (12,))
        # With R = 1 + 0.2 cos(theta) near the midplane, R ranges over ~[0.8, 1.2].
        # The 0.05 inset takes ~5% of that span off each end.
        span = 1.2 - 0.8
        expected_inset = max(0.05 * span, 0.01)
        self.assertGreaterEqual(radii[0], 0.8 + 0.9 * expected_inset)
        self.assertLessEqual(radii[-1], 1.2 - 0.9 * expected_inset)
        self.assertTrue(np.all(np.diff(radii) > 0))

    def test_extended_surface_seed_radii_spans_extended_surface_without_mutating_input(
        self,
    ):
        topology_module = load_topology_scorer_module()

        class _Surface:
            def __init__(self, delta=0.0):
                self.delta = float(delta)
                self.extend_calls = []

            def copy(self):
                return _Surface(self.delta)

            def extend_via_normal(self, distance):
                self.extend_calls.append(float(distance))
                self.delta += float(distance)

            def gamma(self):
                radii = np.array([1.0 - self.delta, 1.0 + self.delta], dtype=float)
                return np.array(
                    [
                        [[radii[0], 0.0, 0.0], [radii[1], 0.0, 0.0]],
                    ],
                    dtype=float,
                )

        surface = _Surface(delta=0.2)
        radii = topology_module.extended_surface_seed_radii(
            surface,
            5,
            extend_distance=0.05,
        )

        self.assertEqual(radii.shape, (5,))
        self.assertAlmostEqual(radii[0], 0.75)
        self.assertAlmostEqual(radii[-1], 1.25)
        self.assertTrue(np.all(np.diff(radii) > 0))
        self.assertEqual(surface.extend_calls, [])

        contract = topology_module.build_extended_surface_seed_contract(
            5,
            0.05,
            radii,
        )
        self.assertEqual(contract["mode"], "extended_surface_radial_sweep")
        self.assertEqual(contract["nfieldlines"], 5)
        self.assertAlmostEqual(contract["extend_distance"], 0.05)
        self.assertEqual(
            contract["radial_sampling_source"], "global_extended_surface_bounds"
        )
        self.assertAlmostEqual(contract["r_min_seed"], 0.75)
        self.assertAlmostEqual(contract["r_max_seed"], 1.25)

    def test_topology_seed_selector_uses_extended_surface_contract(self):
        topology_module = load_topology_scorer_module()

        class _Surface:
            def copy(self):
                return self

            def extend_via_normal(self, distance):
                self.distance = float(distance)

            def gamma(self):
                distance = getattr(self, "distance", 0.0)
                radii = np.array([1.0 - distance, 1.0 + distance], dtype=float)
                return np.array([[[radii[0], 0.0, 0.0], [radii[1], 0.0, 0.0]]])

        radii, contract = topology_module.topology_seed_radii_and_contract(
            _Surface(),
            4,
            seed_mode=topology_module.SEED_MODE_EXTENDED_SURFACE,
            extend_distance=0.07,
        )

        self.assertEqual(contract["mode"], "extended_surface_radial_sweep")
        self.assertEqual(radii.shape, (4,))
        self.assertAlmostEqual(contract["extend_distance"], 0.07)

    def test_extended_surface_seed_radii_clones_real_surface_xyztensorfourier(self):
        topology_module = load_topology_scorer_module()
        from simsopt.geo import SurfaceXYZTensorFourier

        surface = SurfaceXYZTensorFourier(
            nfp=5,
            stellsym=True,
            mpol=2,
            ntor=1,
            quadpoints_phi=np.linspace(0.0, 1.0 / 5.0, 9, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 11, endpoint=False),
        )
        dofs = surface.get_dofs().copy()
        dofs[0] = 1.23
        surface.set_dofs(dofs)
        surface.fix(0)
        original_x = np.asarray(surface.x, dtype=float).copy()
        original_full_x = np.asarray(surface.get_dofs(), dtype=float).copy()
        original_gamma = surface.gamma().copy()

        radii = topology_module.extended_surface_seed_radii(
            surface,
            8,
            extend_distance=0.02,
        )

        self.assertEqual(radii.shape, (8,))
        self.assertTrue(np.all(np.diff(radii) > 0))
        self.assertLess(original_x.size, original_full_x.size)
        clone = topology_module._clone_surface_for_extension(surface)
        np.testing.assert_allclose(clone.get_dofs(), original_full_x)
        np.testing.assert_allclose(np.asarray(surface.x, dtype=float), original_x)
        np.testing.assert_allclose(
            np.asarray(surface.get_dofs(), dtype=float), original_full_x
        )
        np.testing.assert_allclose(surface.gamma(), original_gamma)

    def test_prepare_topology_field_auto_policy_switches_at_threshold(self):
        topology_module = load_topology_scorer_module()

        class _Surface:
            nfp = 5
            stellsym = True

            def gamma(self):
                return np.array(
                    [
                        [[1.0, 0.0, 0.1], [1.1, 0.0, -0.1]],
                        [[0.9, 0.1, 0.05], [1.05, -0.1, -0.05]],
                    ],
                    dtype=float,
                )

        class _BField:
            def __init__(self):
                self.points = None

            def set_points(self, points):
                self.points = np.asarray(points, dtype=float)

            def B(self):
                assert self.points is not None
                return np.ones((self.points.shape[0], 3), dtype=float)

        class _InterpolatedField:
            def __init__(
                self,
                source_field,
                degree,
                rrange,
                phirange,
                zrange,
                extrapolate,
                *,
                nfp,
                stellsym,
            ):
                self.source_field = source_field
                self.degree = degree
                self.rrange = rrange
                self.phirange = phirange
                self.zrange = zrange
                self.extrapolate = extrapolate
                self.nfp = nfp
                self.stellsym = stellsym
                self.points = None

            def set_points(self, points):
                self.points = np.asarray(points, dtype=float)

            def B(self):
                assert self.points is not None
                return np.ones((self.points.shape[0], 3), dtype=float)

        surface = _Surface()
        native_field = _BField()
        below_threshold_field, below_threshold_model = (
            topology_module.prepare_topology_field(
                surface,
                native_field,
                49.9,
                field_policy="auto",
            )
        )
        self.assertIs(below_threshold_field, native_field)
        self.assertEqual(below_threshold_model["selected_mode"], "native")
        self.assertEqual(below_threshold_model["reason"], "below_threshold")

        with patch("simsopt.field.InterpolatedField", _InterpolatedField):
            threshold_field = _BField()
            interpolated_field, interpolated_model = (
                topology_module.prepare_topology_field(
                    surface,
                    threshold_field,
                    50.0,
                    field_policy="auto",
                    interpolation_grid={
                        "degree": 5,
                        "nr": 10,
                        "nphi": 11,
                        "nz": 12,
                    },
                )
            )

        self.assertIsInstance(interpolated_field, _InterpolatedField)
        required_phirange = [0.0, (2.0 * np.pi) / float(surface.nfp)]
        self.assertEqual(interpolated_model["selected_mode"], "interpolated")
        self.assertEqual(interpolated_model["reason"], "tmax_threshold")
        self.assertEqual(
            interpolated_model["grid"],
            {"degree": 5, "nr": 10, "nphi": 11, "nz": 12},
        )
        self.assertEqual(interpolated_model["rrange"], list(interpolated_field.rrange))
        self.assertEqual(
            interpolated_model["phirange"], list(interpolated_field.phirange)
        )
        self.assertEqual(interpolated_model["required_phirange"], required_phirange)
        self.assertEqual(interpolated_model["zrange"], list(interpolated_field.zrange))
        self.assertTrue(interpolated_model["extrapolate"])
        self.assertEqual(
            interpolated_field._topology_interpolation_ranges,
            (
                interpolated_field.rrange,
                interpolated_field.phirange,
                interpolated_field.zrange,
            ),
        )
        self.assertFalse(interpolated_model["interpolation_covers_trace_domain"])
        trace_domain = interpolated_model["trace_domain"]
        self.assertGreater(
            interpolated_model["rrange"][0], trace_domain["required_rmin"]
        )
        self.assertLess(interpolated_model["rrange"][1], trace_domain["required_rmax"])
        self.assertGreaterEqual(
            interpolated_model["zrange"][1], trace_domain["required_zmax"]
        )
        self.assertEqual(interpolated_model["max_abs_error"], 0.0)
        self.assertEqual(interpolated_model["mean_abs_error"], 0.0)
        self.assertEqual(interpolated_model["max_rel_error"], 0.0)

    def test_prepare_topology_field_records_metric_domain_exceeding_interpolation(self):
        topology_module = load_topology_scorer_module()

        class _Surface:
            nfp = 5
            stellsym = True

            def gamma(self):
                return np.array(
                    [
                        [[1.0, 0.0, 0.1], [1.1, 0.0, -0.1]],
                        [[0.9, 0.1, 0.05], [1.05, -0.1, -0.05]],
                    ],
                    dtype=float,
                )

        class _BField:
            def __init__(self):
                self.points = None

            def set_points(self, points):
                self.points = np.asarray(points, dtype=float)

            def B(self):
                assert self.points is not None
                return np.ones((self.points.shape[0], 3), dtype=float)

        class _InterpolatedField:
            def __init__(
                self,
                source_field,
                degree,
                rrange,
                phirange,
                zrange,
                extrapolate,
                *,
                nfp,
                stellsym,
            ):
                self.source_field = source_field
                self.rrange = rrange
                self.phirange = phirange
                self.zrange = zrange
                self.extrapolate = extrapolate
                self.points = None

            def set_points(self, points):
                self.points = np.asarray(points, dtype=float)

            def B(self):
                assert self.points is not None
                return np.ones((self.points.shape[0], 3), dtype=float)

        surface = _Surface()
        trace_domain = topology_module.surface_trace_domain(
            surface,
            seed_radii=np.array([0.80, 1.20]),
        )
        old_rmin, old_rmax, _old_zmax = topology_module.padded_bounds(
            trace_domain.surface_rmin,
            trace_domain.surface_rmax,
            trace_domain.surface_zmax,
        )
        self.assertGreater(trace_domain.required_rmax, old_rmax)
        self.assertLess(trace_domain.required_rmin, old_rmin)

        with patch("simsopt.field.InterpolatedField", _InterpolatedField):
            interpolated_field, interpolated_model = (
                topology_module.prepare_topology_field(
                    surface,
                    _BField(),
                    50.0,
                    field_policy="auto",
                    trace_domain=trace_domain,
                )
            )

        self.assertIsInstance(interpolated_field, _InterpolatedField)
        self.assertEqual(interpolated_model["selected_mode"], "interpolated")
        self.assertFalse(interpolated_model["interpolation_covers_trace_domain"])
        self.assertGreater(interpolated_field.rrange[0], trace_domain.required_rmin)
        self.assertLess(interpolated_field.rrange[1], trace_domain.required_rmax)
        self.assertGreaterEqual(
            interpolated_field.zrange[1], trace_domain.required_zmax
        )
        self.assertTrue(interpolated_model["extrapolate"])

    def test_prepare_topology_field_rejects_uncovered_explicit_interpolation_domain(
        self,
    ):
        topology_module = load_topology_scorer_module()

        class _Surface:
            nfp = 5
            stellsym = True

            def gamma(self):
                return np.array(
                    [
                        [[1.0, 0.0, 0.1], [1.1, 0.0, -0.1]],
                        [[0.9, 0.1, 0.05], [1.05, -0.1, -0.05]],
                    ],
                    dtype=float,
                )

        class _BField:
            pass

        surface = _Surface()
        native_field, native_model = topology_module.prepare_topology_field(
            surface,
            _BField(),
            50.0,
            field_policy="auto",
            interpolation_grid={
                "rrange": (0.95, 1.02, 10),
                "zrange": (0.0, 0.05, 10),
            },
        )

        self.assertIsInstance(native_field, _BField)
        self.assertEqual(native_model["selected_mode"], "native")
        self.assertEqual(native_model["reason"], "trace_domain_not_covered")
        self.assertFalse(native_model["interpolation_covers_trace_domain"])

        with self.assertRaisesRegex(ValueError, "does not cover"):
            topology_module.prepare_topology_field(
                surface,
                _BField(),
                50.0,
                field_policy="always",
                interpolation_grid={
                    "rrange": (0.95, 1.02, 10),
                    "zrange": (0.0, 0.05, 10),
                },
            )

        trace_domain = topology_module.surface_trace_domain(surface)
        required_phirange = [0.0, (2.0 * np.pi) / float(surface.nfp)]
        narrow_phi_grid = {
            "rrange": (
                trace_domain.required_rmin - 0.01,
                trace_domain.required_rmax + 0.01,
                10,
            ),
            "phirange": (0.0, 0.1, 10),
            "zrange": (0.0, trace_domain.required_zmax + 0.01, 10),
        }
        native_field, native_model = topology_module.prepare_topology_field(
            surface,
            _BField(),
            50.0,
            field_policy="auto",
            interpolation_grid=narrow_phi_grid,
        )

        self.assertIsInstance(native_field, _BField)
        self.assertEqual(native_model["selected_mode"], "native")
        self.assertEqual(native_model["reason"], "trace_domain_not_covered")
        self.assertFalse(native_model["interpolation_covers_trace_domain"])
        self.assertEqual(native_model["phirange"], [0.0, 0.1, 10])
        self.assertEqual(native_model["required_phirange"], required_phirange)

        with self.assertRaisesRegex(ValueError, "does not cover"):
            topology_module.prepare_topology_field(
                surface,
                _BField(),
                50.0,
                field_policy="always",
                interpolation_grid=narrow_phi_grid,
            )

    def test_poincare_field_policy_env_validation(self):
        module = load_poincare_surfaces_module()

        self.assertEqual(module.resolve_poincare_field_policy({}), "auto")
        self.assertEqual(
            module.resolve_poincare_field_policy({"POINCARE_FIELD_POLICY": "auto"}),
            "auto",
        )
        self.assertEqual(
            module.resolve_poincare_field_policy({"POINCARE_FIELD_POLICY": "never"}),
            "never",
        )
        with self.assertRaisesRegex(ValueError, "Unsupported POINCARE_FIELD_POLICY"):
            module.resolve_poincare_field_policy({"POINCARE_FIELD_POLICY": "sometimes"})

    def test_poincare_render_mode_env_validation(self):
        module = load_poincare_surfaces_module()

        self.assertEqual(
            module.resolve_poincare_render_mode_names({}),
            ("validation", "diagnostic", "default"),
        )
        self.assertEqual(
            module.resolve_poincare_render_mode_names(
                {"POINCARE_RENDER_MODES": "default"}
            ),
            ("default",),
        )
        self.assertEqual(
            module.resolve_poincare_render_mode_names(
                {"POINCARE_RENDER_MODES": "diagnostic,default"}
            ),
            ("diagnostic", "default"),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported POINCARE_RENDER_MODES"):
            module.resolve_poincare_render_mode_names(
                {"POINCARE_RENDER_MODES": "strict"}
            )
        with self.assertRaisesRegex(ValueError, "must not repeat"):
            module.resolve_poincare_render_mode_names(
                {"POINCARE_RENDER_MODES": "default,default"}
            )
        with self.assertRaisesRegex(ValueError, "comma-separated"):
            module.resolve_poincare_render_mode_names(
                {"POINCARE_RENDER_MODES": "default,"}
            )

    def test_poincare_render_modes_preserve_metric_contracts(self):
        module = load_poincare_surfaces_module()

        class _Surface:
            nfp = 5
            stellsym = True

            def __init__(self, extension=0.0):
                self.extension = float(extension)

            def copy(self):
                return _Surface(self.extension)

            def extend_via_normal(self, distance):
                self.extension += float(distance)

            def gamma(self):
                theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
                radius = 0.2 + self.extension
                R = 1.0 + radius * np.cos(theta)
                Z = 0.1 * np.sin(theta)
                return np.array([np.column_stack([R, np.zeros_like(R), Z])])

            def cross_section(self, phi=0.0, thetas=512):
                theta = np.linspace(0.0, 2.0 * np.pi, int(thetas), endpoint=False)
                radius = 0.2 + self.extension
                R = 1.0 + radius * np.cos(theta)
                Z = 0.1 * np.sin(theta)
                phi_abs = 2.0 * np.pi * float(phi)
                return np.column_stack([R * np.cos(phi_abs), R * np.sin(phi_abs), Z])

        guarded_stopping_criteria = [object(), object()]
        default_stopping_criteria = [object()]
        modes, field_domain = module.build_poincare_render_modes(
            _Surface(),
            6,
            seed_inset_fraction=0.05,
            default_extend_distance=0.05,
            guarded_stopping_criteria=guarded_stopping_criteria,
            guarded_stop_labels=["surface_exit", "max_r_guardrail", "iteration_limit"],
            default_stopping_criteria=default_stopping_criteria,
            default_stop_labels=["max_r_guardrail", "iteration_limit"],
        )

        self.assertEqual(
            [mode["mode"] for mode in modes], ["validation", "diagnostic", "default"]
        )
        self.assertIs(modes[0]["stopping_criteria"], guarded_stopping_criteria)
        self.assertIs(modes[1]["stopping_criteria"], guarded_stopping_criteria)
        self.assertIs(modes[2]["stopping_criteria"], default_stopping_criteria)
        self.assertIn("surface_exit", modes[1]["stop_labels"])
        self.assertNotIn("surface_exit", modes[2]["stop_labels"])
        self.assertEqual(modes[0]["trace_semantics"], "surface_exit_guarded")
        self.assertEqual(modes[2]["trace_semantics"], "baseline_wander")
        self.assertEqual(modes[0]["field_policy"], "configured")
        self.assertEqual(modes[2]["field_policy"], "native")
        self.assertEqual(modes[0]["seed_contract"]["mode"], "midplane_radial_sweep")
        self.assertEqual(modes[1]["seed_contract"]["mode"], "midplane_radial_sweep")
        self.assertEqual(
            modes[2]["seed_contract"]["mode"],
            "extended_surface_radial_sweep",
        )
        self.assertEqual(field_domain, modes[2]["trace_domain"])
        self.assertLessEqual(
            field_domain.required_rmin, modes[0]["trace_domain"].required_rmin
        )
        self.assertGreaterEqual(
            field_domain.required_rmax, modes[0]["trace_domain"].required_rmax
        )
        self.assertGreaterEqual(
            field_domain.required_zmax, modes[0]["trace_domain"].required_zmax
        )
        self.assertAlmostEqual(field_domain.surface_rmin, 0.75)
        self.assertAlmostEqual(field_domain.surface_rmax, 1.25)
        self.assertAlmostEqual(field_domain.stopping_rmin, 0.75 * 0.95)
        self.assertAlmostEqual(field_domain.stopping_rmax, 1.25 * 1.05)
        self.assertAlmostEqual(field_domain.stopping_zmax, 0.1 * 1.05)

    def test_select_poincare_render_modes_keeps_requested_order(self):
        module = load_poincare_surfaces_module()
        render_modes = [
            {"mode": "validation"},
            {"mode": "diagnostic"},
            {"mode": "default"},
        ]

        selected = module.select_poincare_render_modes(
            render_modes,
            ("default", "validation"),
        )

        self.assertEqual([mode["mode"] for mode in selected], ["default", "validation"])
        with self.assertRaisesRegex(ValueError, "not built"):
            module.select_poincare_render_modes(render_modes, ("strict",))

    def test_partial_poincare_modes_reject_stale_unselected_metrics(self):
        module = load_poincare_surfaces_module()
        all_modes = [
            {"mode": "validation", "metrics_suffix": "_validation"},
            {"mode": "diagnostic", "metrics_suffix": "_diagnostic"},
            {"mode": "default", "metrics_suffix": "_default"},
        ]
        selected_modes = [all_modes[2]]
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(
                os.path.join(tmpdir, "PoincareMetrics_opt_validation.json"),
                "w",
                encoding="utf-8",
            ) as fh:
                fh.write("{}")

            with self.assertRaisesRegex(FileExistsError, "stale metrics"):
                module.assert_no_stale_unselected_metrics(
                    tmpdir,
                    "opt",
                    all_modes,
                    selected_modes,
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            module.assert_no_stale_unselected_metrics(
                tmpdir,
                "opt",
                all_modes,
                selected_modes,
            )

    def test_prepare_poincare_fields_keeps_default_native_wander(self):
        module = load_poincare_surfaces_module()
        calls = []

        class _Field:
            pass

        class _Surface:
            pass

        def fake_prepare_topology_field(
            surf,
            bs,
            tmax_fl,
            *,
            field_policy,
            interpolation_grid,
            trace_domain,
        ):
            del surf, bs, tmax_fl
            calls.append(
                {
                    "field_policy": field_policy,
                    "interpolation_grid": interpolation_grid,
                    "trace_domain": trace_domain,
                }
            )
            return _Field(), {
                "selected_mode": "native"
                if field_policy == "never"
                else "interpolated",
                "reason": "explicit_never"
                if field_policy == "never"
                else "tmax_threshold",
            }

        guarded_domain = object()
        default_domain = object()
        render_modes = [
            {
                "mode": "validation",
                "field_key": "guarded",
                "field_policy": "configured",
                "trace_domain": guarded_domain,
                "trace_semantics": "surface_exit_guarded",
            },
            {
                "mode": "diagnostic",
                "field_key": "guarded",
                "field_policy": "configured",
                "trace_domain": guarded_domain,
                "trace_semantics": "surface_exit_guarded",
            },
            {
                "mode": "default",
                "field_key": "baseline_wander",
                "field_policy": "native",
                "trace_domain": default_domain,
                "trace_semantics": "baseline_wander",
            },
        ]

        with patch.object(
            module,
            "prepare_topology_field",
            side_effect=fake_prepare_topology_field,
        ):
            fields_by_mode, field_models_by_mode = module.prepare_poincare_fields(
                _Surface(),
                _Field(),
                7000,
                render_modes=render_modes,
                field_model_policy="auto",
                interpolation_grid={"degree": 3, "nr": 40, "nphi": 40, "nz": 20},
            )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["field_policy"], "auto")
        self.assertIs(calls[0]["trace_domain"], guarded_domain)
        self.assertEqual(calls[1]["field_policy"], "never")
        self.assertIs(calls[1]["trace_domain"], default_domain)
        self.assertIs(fields_by_mode["validation"], fields_by_mode["diagnostic"])
        self.assertIsNot(fields_by_mode["validation"], fields_by_mode["default"])
        self.assertEqual(
            field_models_by_mode["validation"]["selected_mode"],
            "interpolated",
        )
        self.assertEqual(field_models_by_mode["default"]["selected_mode"], "native")
        self.assertEqual(
            field_models_by_mode["default"]["poincare_trace_semantics"],
            "baseline_wander",
        )

    def test_compute_topology_transport_diagnostics_reports_surface_structure(self):
        topology_module = load_topology_scorer_module()

        class _Surface:
            def gamma(self):
                return np.array(
                    [
                        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                        [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                    ],
                    dtype=float,
                )

        class _Field:
            def set_points(self, points):
                self._points = np.asarray(points, dtype=float)

            def AbsB(self):
                return self._points[:, 0] + 1.0

        diagnostics = topology_module.compute_topology_transport_diagnostics(
            _Surface(),
            _Field(),
        )

        self.assertEqual(
            diagnostics["schema_version"],
            "single_stage_topology_transport_diagnostics_v1",
        )
        self.assertEqual(diagnostics["status"], "partial")
        self.assertEqual(
            diagnostics["surface_field_structure"]["status"],
            "evaluated",
        )
        self.assertEqual(
            diagnostics["surface_field_structure"]["grid_shape"],
            [2, 2],
        )
        self.assertAlmostEqual(
            diagnostics["surface_field_structure"]["modB_min"],
            2.0,
        )
        self.assertAlmostEqual(
            diagnostics["surface_field_structure"]["modB_max"],
            5.0,
        )
        self.assertAlmostEqual(
            diagnostics["surface_field_structure"]["mirror_ratio"],
            2.5,
        )
        self.assertAlmostEqual(
            diagnostics["surface_field_structure"][
                "effective_inverse_aspect_ratio_epsilon"
            ],
            3.0 / 7.0,
        )
        self.assertEqual(diagnostics["gamma_c"]["status"], "unavailable")
        self.assertEqual(
            diagnostics["effective_ripple"]["aliases"],
            ["epsilon_eff"],
        )

    def test_topology_fidelity_report_summarizes_tier_agreement(self):
        ladder_module = load_topology_fidelity_ladder_module()

        report = ladder_module.build_topology_fidelity_report(
            [
                {
                    "label": "case_a",
                    "cheap": {"passed": True, "confinement_score": 0.80},
                    "medium": {"passed": True, "confinement_score": 0.82},
                    "strict": {"passed": False, "confinement_score": 0.20},
                },
                {
                    "label": "case_b",
                    "cheap": {"passed": False, "confinement_score": 0.30},
                    "medium": {"passed": False, "confinement_score": 0.35},
                    "strict": {"passed": True, "confinement_score": 0.70},
                },
                {
                    "label": "case_c",
                    "cheap": {"passed": True, "confinement_score": 0.60},
                    "medium": {"passed": True, "confinement_score": 0.62},
                    "strict": {"passed": True, "confinement_score": 0.65},
                },
            ]
        )

        cheap_vs_strict = report["agreements"]["cheap_vs_strict"]
        medium_vs_strict = report["agreements"]["medium_vs_strict"]
        self.assertEqual(cheap_vs_strict["false_pass_count"], 1)
        self.assertEqual(cheap_vs_strict["false_reject_count"], 1)
        self.assertEqual(cheap_vs_strict["false_pass_labels"], ["case_a"])
        self.assertEqual(cheap_vs_strict["false_reject_labels"], ["case_b"])
        self.assertEqual(medium_vs_strict["false_pass_count"], 1)
        self.assertEqual(medium_vs_strict["false_reject_count"], 1)
        self.assertIsNotNone(cheap_vs_strict["spearman_rank_correlation"])
        self.assertEqual(report["schema_version"], "topology_fidelity_ladder_v2")
        cheap_spec = report["tier_specs"]["cheap"]
        self.assertEqual(
            cheap_spec["seed_mode"],
            "midplane_radial_sweep",
        )
        self.assertEqual(cheap_spec["inset_fraction"], 0.05)
        self.assertEqual(report["tier_specs"]["medium"]["field_policy"], "auto")

    def test_topology_gate_rejection_increment_scales_with_deficit(self):
        module = self.load_module()

        status = {
            "enabled": True,
            "survival_fraction": 0.5,
            "survival_threshold": 0.75,
        }

        self.assertAlmostEqual(module.topology_gate_deficit(status), 0.25)
        self.assertAlmostEqual(
            module.topology_gate_rejection_increment(42.0, status, 4.0),
            84.0,
        )

    def test_fun_rejects_candidate_on_topology_gate_failure(self):
        module = self.load_module()

        last_J = 42.0
        last_dJ = np.array([1.0, -2.0, 3.0, -4.0, 5.0])

        class _Surface:
            x = np.ones(3)

            def is_self_intersecting(self):
                return False

            def volume(self):
                return 1.0

            def gamma(self):
                return np.zeros((1, 1, 3))

        class _BoozerSurface:
            surface = _Surface()
            res = {"success": True, "iota": TEST_IOTA, "G": TEST_G0}

            def run_code(self, iota, G):
                return self.res

        class _JF:
            x = np.zeros(5)

        surface_data = [
            {"boozer_surface": _BoozerSurface()},
            {"boozer_surface": _BoozerSurface()},
        ]
        module.run_dict = {
            "x_prev": np.zeros(5),
            "lscount": 0,
            "surface_state": {
                "sdofs": [np.ones(3), np.ones(3)],
                "iota": [TEST_IOTA, TEST_IOTA],
                "G": [TEST_G0, TEST_G0],
            },
            "J": last_J,
            "dJ": last_dJ.copy(),
            "accepted_iterations": 0,
            "accepted_x": np.zeros(5),
        }
        module.surface_data = surface_data
        module.outer_surface_data = surface_data[-1]
        module.surface_iota_terms = [
            SimpleNamespace(J=lambda: TEST_IOTA),
            SimpleNamespace(J=lambda: TEST_IOTA),
        ]
        module.VV = object()
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.JF = _JF()
        module.bs = object()
        module.nonQSs = []
        module.brs = []
        module.Jiota = object()
        module.IOTAS_WEIGHT = 1.0
        module.JCurveLength = object()
        module.LENGTH_WEIGHT = 1.0
        module.JCurveCurve = object()
        module.CC_WEIGHT = 1.0
        module.JCurveSurface = object()
        module.CS_WEIGHT = 1.0
        module.JCurvature = object()
        module.CURVATURE_WEIGHT = 1.0
        module.RES_WEIGHT = 1.0
        module.MULTISURFACE_RAMP_ITERATIONS = 0
        module.INNER_SURFACE_INITIAL_WEIGHT = 1.0
        module.TOPOLOGY_GATE_FIELDLINES = 4
        module.TOPOLOGY_GATE_TMAX = 2.0
        module.TOPOLOGY_GATE_TOL = 1e-7
        module.TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.25
        module.TOPOLOGY_GATE_PENALTY_SCALE = 4.0

        with (
            patch.object(
                module,
                "solve_surface_stack_at_dofs",
                return_value={
                    "success": True,
                    "solve_success": [True, True],
                    "self_intersections": [False, False],
                    "volumes_ordered": True,
                    "gap_ok": True,
                    "nesting_ok": True,
                    "adjacent_gaps": [0.01],
                    "outer_vessel_gap": 0.1,
                    "bad_nesting_phis": [],
                },
            ),
            patch.object(
                module,
                "evaluate_total_objective",
                return_value={
                    "total": 7.0,
                    "grad": np.arange(5, dtype=float),
                    "surface_weights": np.array([1.0, 1.0]),
                },
            ),
            patch.object(module, "restore_surface_states") as restore_mock,
            patch.object(
                module,
                "evaluate_topology_gate",
                return_value={
                    "enabled": True,
                    "success": False,
                    "state": "modeled_infeasible",
                    "broken": False,
                    "nfieldlines": 4,
                    "survived_lines": 0,
                    "survival_fraction": 0.0,
                    "survival_threshold": 0.25,
                    "tmax": 2.0,
                    "tol": 1e-7,
                    "stop_reason_counts": {"surface_exit": 4},
                    "first_exit_time": 0.4,
                    "first_exit_angle": 0.2,
                    "first_exit_reason": "surface_exit",
                    "evaluation_error": None,
                    "evaluation_error_type": None,
                },
            ),
        ):
            J_out, dJ_out = module.fun(np.ones(5))

        self.assertEqual(J_out, 126.0)
        np.testing.assert_array_equal(dJ_out, last_dJ)
        restore_mock.assert_called_once()
        self.assertEqual(
            module.run_dict["topology_gate_status"]["state"], "modeled_infeasible"
        )
        self.assertEqual(module.run_dict["topology_gate_rejects"], 1)
        self.assertEqual(module.run_dict["invalid_state_rejects_total"], 0)

    def test_fun_rejects_candidate_on_broken_topology_evaluation(self):
        module = self.load_module()

        last_J = 42.0
        last_dJ = np.array([1.0, -2.0, 3.0, -4.0, 5.0])

        class _Surface:
            x = np.ones(3)

            def is_self_intersecting(self):
                return False

            def volume(self):
                return 1.0

            def gamma(self):
                return np.zeros((1, 1, 3))

        class _BoozerSurface:
            surface = _Surface()
            res = {"success": True, "iota": TEST_IOTA, "G": TEST_G0}

            def run_code(self, iota, G):
                return self.res

        class _JF:
            x = np.zeros(5)

        surface_data = [
            {"boozer_surface": _BoozerSurface()},
            {"boozer_surface": _BoozerSurface()},
        ]
        module.run_dict = {
            "x_prev": np.zeros(5),
            "lscount": 0,
            "surface_state": {
                "sdofs": [np.ones(3), np.ones(3)],
                "iota": [TEST_IOTA, TEST_IOTA],
                "G": [TEST_G0, TEST_G0],
            },
            "J": last_J,
            "dJ": last_dJ.copy(),
            "accepted_iterations": 0,
            "accepted_x": np.zeros(5),
        }
        module.surface_data = surface_data
        module.outer_surface_data = surface_data[-1]
        module.surface_iota_terms = [
            SimpleNamespace(J=lambda: TEST_IOTA),
            SimpleNamespace(J=lambda: TEST_IOTA),
        ]
        module.VV = object()
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.JF = _JF()
        module.bs = object()
        module.nonQSs = []
        module.brs = []
        module.Jiota = object()
        module.IOTAS_WEIGHT = 1.0
        module.JCurveLength = object()
        module.LENGTH_WEIGHT = 1.0
        module.JCurveCurve = object()
        module.CC_WEIGHT = 1.0
        module.JCurveSurface = object()
        module.CS_WEIGHT = 1.0
        module.JCurvature = object()
        module.CURVATURE_WEIGHT = 1.0
        module.RES_WEIGHT = 1.0
        module.MULTISURFACE_RAMP_ITERATIONS = 0
        module.INNER_SURFACE_INITIAL_WEIGHT = 1.0
        module.TOPOLOGY_GATE_FIELDLINES = 4
        module.TOPOLOGY_GATE_TMAX = 2.0
        module.TOPOLOGY_GATE_TOL = 1e-7
        module.TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.25
        module.TOPOLOGY_GATE_PENALTY_SCALE = 4.0

        with (
            patch.object(
                module,
                "solve_surface_stack_at_dofs",
                return_value={
                    "success": True,
                    "solve_success": [True, True],
                    "self_intersections": [False, False],
                    "volumes_ordered": True,
                    "gap_ok": True,
                    "nesting_ok": True,
                    "adjacent_gaps": [0.01],
                    "outer_vessel_gap": 0.1,
                    "bad_nesting_phis": [],
                },
            ),
            patch.object(
                module,
                "evaluate_total_objective",
                return_value={
                    "total": 7.0,
                    "grad": np.arange(5, dtype=float),
                    "surface_weights": np.array([1.0, 1.0]),
                },
            ),
            patch.object(module, "restore_surface_states") as restore_mock,
            patch.object(
                module,
                "evaluate_topology_gate",
                side_effect=RuntimeError("trace exploded"),
            ),
        ):
            J_out, dJ_out = module.fun(np.ones(5))

        self.assertEqual(J_out, last_J * 2.0)
        np.testing.assert_array_equal(dJ_out, last_dJ)
        restore_mock.assert_called_once()
        self.assertEqual(module.run_dict["topology_gate_status"]["state"], "broken")
        self.assertTrue(module.run_dict["topology_gate_status"]["broken"])
        self.assertIn(
            "trace exploded",
            module.run_dict["topology_gate_status"]["evaluation_error"],
        )
        self.assertEqual(
            module.run_dict["topology_gate_status"]["evaluation_error_type"],
            "RuntimeError",
        )
        self.assertEqual(module.run_dict["topology_gate_rejects"], 0)
        self.assertEqual(module.run_dict["invalid_state_rejects_total"], 1)

    def test_build_surface_configs_two_surface_mode_derives_inner_target_volume(self):
        module = self.load_module()

        class _FakeRZSurface:
            def __init__(self, label):
                self.label = label
                self._major_radius = 2.0
                self._volume = 10.0 * label
                self.nfp = 5
                self._dofs = np.array([1.0])

            def major_radius(self):
                return self._major_radius

            def get_dofs(self):
                return self._dofs.copy()

            def set_dofs(self, dofs):
                dofs = np.asarray(dofs)
                scale = dofs[0] / self._dofs[0]
                self._dofs = dofs
                self._major_radius *= scale
                self._volume *= scale**3

            def volume(self):
                return self._volume

        fake_factory = SimpleNamespace(
            from_wout=lambda *_args, **kwargs: _FakeRZSurface(kwargs["s"]),
        )

        with patch.object(module, "SurfaceRZFourier", fake_factory):
            configs = module.build_surface_configs(
                "dummy.nc",
                nphi=11,
                ntheta=13,
                seed_label=0.25,
                major_radius=1.0,
                outer_target_volume=0.10,
                num_surfaces=2,
                inner_surface_ratio=0.8,
            )

        self.assertEqual([config["name"] for config in configs], ["inner", "outer"])
        self.assertAlmostEqual(configs[0]["seed_label"], 0.20)
        self.assertAlmostEqual(configs[1]["seed_label"], 0.25)
        self.assertAlmostEqual(configs[0]["target_volume"], 0.08)
        self.assertAlmostEqual(configs[1]["target_volume"], 0.10)

    def test_build_surface_configs_single_surface_mode_keeps_outer_only_contract(self):
        module = self.load_module()

        class _FakeRZSurface:
            def __init__(self, label):
                self.label = label
                self._major_radius = 2.0
                self._volume = 10.0 * label
                self.nfp = 5
                self._dofs = np.array([1.0])

            def major_radius(self):
                return self._major_radius

            def get_dofs(self):
                return self._dofs.copy()

            def set_dofs(self, dofs):
                dofs = np.asarray(dofs)
                scale = dofs[0] / self._dofs[0]
                self._dofs = dofs
                self._major_radius *= scale
                self._volume *= scale**3

            def volume(self):
                return self._volume

        fake_factory = SimpleNamespace(
            from_wout=lambda *_args, **kwargs: _FakeRZSurface(kwargs["s"]),
        )

        with patch.object(module, "SurfaceRZFourier", fake_factory):
            configs = module.build_surface_configs(
                "dummy.nc",
                nphi=11,
                ntheta=13,
                seed_label=0.25,
                major_radius=1.0,
                outer_target_volume=0.10,
                num_surfaces=1,
                inner_surface_ratio=0.8,
            )

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["name"], "outer")
        self.assertAlmostEqual(configs[0]["seed_label"], 0.25)
        self.assertAlmostEqual(configs[0]["target_volume"], 0.10)

    def test_build_surface_configs_legacy_rejects_more_than_two_surfaces(self):
        module = self.load_module()

        with self.assertRaisesRegex(ValueError, "Legacy build_surface_configs"):
            module.build_surface_configs(
                "dummy.nc",
                nphi=11,
                ntheta=13,
                seed_label=0.25,
                major_radius=1.0,
                outer_target_volume=0.10,
                num_surfaces=3,
                inner_surface_ratio=0.8,
            )

    def test_build_surface_configs_for_contract_builds_published_fixed_stack(self):
        module = self.load_module()

        class _FakeRZSurface:
            def __init__(self, label):
                self.label = label
                self._major_radius = 2.0
                self._volume = 10.0 * label
                self.nfp = 5
                self._dofs = np.array([1.0])

            def major_radius(self):
                return self._major_radius

            def get_dofs(self):
                return self._dofs.copy()

            def set_dofs(self, dofs):
                dofs = np.asarray(dofs)
                scale = dofs[0] / self._dofs[0]
                self._dofs = dofs
                self._major_radius *= scale
                self._volume *= scale**3

            def volume(self):
                return self._volume

        fake_factory = SimpleNamespace(
            from_wout=lambda *_args, **kwargs: _FakeRZSurface(kwargs["s"]),
        )
        contract = module.resolve_surface_mode_contract(
            SimpleNamespace(
                surface_mode=module.PUBLISHED_MULTISURFACE,
                num_surfaces=1,
                inner_surface_ratio=0.8,
            ),
            warn_on_legacy_mapping=False,
        )

        with patch.object(module, "SurfaceRZFourier", fake_factory):
            configs = module.build_surface_configs_for_contract(
                "dummy.nc",
                nphi=11,
                ntheta=13,
                seed_label=0.25,
                major_radius=1.0,
                outer_target_volume=0.10,
                surface_mode_contract=contract,
            )

        self.assertEqual(
            [config["name"] for config in configs],
            ["inner0", "inner1", "outer"],
        )
        np.testing.assert_allclose(
            [config["seed_label"] for config in configs],
            [0.15, 0.20, 0.25],
        )
        np.testing.assert_allclose(
            [config["target_volume"] for config in configs],
            [0.06, 0.08, 0.10],
        )

    def test_edge_delivered_iota_preset_resolves_edge_surface_stack(self):
        module = self.load_module()

        contract = module.resolve_surface_mode_contract(
            SimpleNamespace(
                surface_mode=module.PUBLISHED_MULTISURFACE,
                num_surfaces=1,
                inner_surface_ratio=0.8,
                published_surface_preset=(
                    module.PUBLISHED_PRESET_EDGE_DELIVERED_IOTA_LANE
                ),
            ),
            warn_on_legacy_mapping=False,
        )

        np.testing.assert_allclose(
            contract.label_fractions,
            [0.70, 0.85, 0.95, 1.0],
        )

    def test_published_stage2_seed_initializes_outer_to_inner_and_returns_storage_order(
        self,
    ):
        module = self.load_module()
        family = importlib.import_module("banana_opt.boozer_surface_family")
        contract = module.resolve_surface_mode_contract(
            SimpleNamespace(
                surface_mode=module.PUBLISHED_MULTISURFACE,
                num_surfaces=1,
                inner_surface_ratio=0.8,
            ),
            warn_on_legacy_mapping=False,
        )
        surface_configs = [
            {
                "name": "inner0",
                "seed_label": 0.15,
                "target_volume": 0.06,
                "initial_surface": SimpleNamespace(name="wout_inner0"),
            },
            {
                "name": "inner1",
                "seed_label": 0.20,
                "target_volume": 0.08,
                "initial_surface": SimpleNamespace(name="wout_inner1"),
            },
            {
                "name": "outer",
                "seed_label": 0.25,
                "target_volume": 0.10,
                "initial_surface": SimpleNamespace(name="wout_outer"),
            },
        ]
        init_calls = []
        contraction_calls = []

        class FakeSurface:
            def __init__(self, name, volume):
                self.name = name
                self._volume = volume

            def volume(self):
                return self._volume

        def fake_initialize_boozer_surface(
            initial_surface,
            _mpol,
            _ntor,
            _bs,
            vol_target,
            _constraint_weight,
            iota,
            G0,
            _boozer_I,
            *,
            initial_surface_guess,
            nfp,
        ):
            init_calls.append(
                {
                    "surface": initial_surface.name,
                    "guess": initial_surface_guess.name,
                    "target_volume": vol_target,
                    "iota": iota,
                    "G": G0,
                    "nfp": nfp,
                }
            )
            solved_name = {
                "stage2_outer": "solved_outer",
                "contracted_solved_outer_to_0.08": "solved_inner1",
                "contracted_solved_inner1_to_0.06": "solved_inner0",
            }[initial_surface.name]
            solved_volume = {
                "solved_outer": 0.10,
                "solved_inner1": 0.08,
                "solved_inner0": 0.06,
            }[solved_name]
            return SimpleNamespace(
                success=True,
                solve_success=True,
                self_intersecting=False,
                solved_iota=float(iota) + 0.01,
                solved_G=float(G0) + 1e-9,
                volume=solved_volume,
                boozer_surface=SimpleNamespace(
                    surface=FakeSurface(solved_name, solved_volume),
                    res={"iota": float(iota) + 0.01, "G": float(G0) + 1e-9},
                ),
            )

        def fake_contract_surface_to_target_volume(previous_surface, target_volume):
            contraction_calls.append((previous_surface.name, target_volume))
            return FakeSurface(
                f"contracted_{previous_surface.name}_to_{target_volume:.2f}",
                target_volume,
            )

        stage2_seed = SimpleNamespace(
            surface=SimpleNamespace(name="stage2_outer"),
            iota=0.21,
            G=0.77,
        )
        with (
            patch.object(
                family,
                "attempt_initialize_boozer_surface",
                side_effect=fake_initialize_boozer_surface,
            ),
            patch.object(
                module,
                "contract_surface_to_target_volume",
                side_effect=fake_contract_surface_to_target_volume,
            ),
            patch.object(module, "cross_sections_are_nested", return_value=(True, [])),
        ):
            surface_data, warm_paths = module.initialize_surface_data_for_contract(
                surface_configs,
                surface_mode_contract=contract,
                mpol=8,
                ntor=6,
                bs=object(),
                constraint_weight=1.0,
                default_iota=0.15,
                default_G=1.0,
                boozer_I=0.0,
                nfp=5,
                stage2_seed_surface=stage2_seed,
                warm_start_surface_stem=None,
            )

        self.assertEqual(
            [call["surface"] for call in init_calls],
            [
                "stage2_outer",
                "contracted_solved_outer_to_0.08",
                "contracted_solved_inner1_to_0.06",
            ],
        )
        self.assertEqual(
            [call["guess"] for call in init_calls],
            [call["surface"] for call in init_calls],
        )
        np.testing.assert_allclose(
            [call["G"] for call in init_calls],
            [0.77, 0.770000001, 0.770000002],
        )
        self.assertTrue(all(np.isfinite(call["G"]) for call in init_calls))
        self.assertEqual(
            contraction_calls,
            [("solved_outer", 0.08), ("solved_inner1", 0.06)],
        )
        self.assertEqual(
            [entry["name"] for entry in surface_data], ["inner0", "inner1", "outer"]
        )
        self.assertEqual(
            [entry["initialization_provenance"] for entry in surface_data],
            [
                "inner1_continuation_inner0",
                "outer_continuation_inner1",
                "stage2_outer_seed",
            ],
        )
        self.assertEqual(warm_paths, [])

    def test_published_multisurface_vacuum_request_normalizes_default_donor_mode(
        self,
    ):
        module = self.load_module()
        contract = module.resolve_surface_mode_contract(
            SimpleNamespace(
                surface_mode=module.PUBLISHED_MULTISURFACE,
                num_surfaces=1,
                inner_surface_ratio=0.8,
            ),
            warn_on_legacy_mapping=False,
        )
        requested_args = SimpleNamespace(
            boozer_I=None,
            plasma_current_A=None,
            finite_current_mode="vacuum",
        )

        finite_current_mode = (
            module.resolve_stage2_finite_current_mode_for_surface_mode(
                {"FINITE_CURRENT_MODE": module.DEFAULT_FINITE_CURRENT_MODE},
                requested_args.finite_current_mode,
                contract,
            )
        )
        settings = module.resolve_plasma_current_settings(
            requested_args,
            finite_current_mode=finite_current_mode,
            default_plasma_current_A=0.0,
            num_surfaces=contract.num_surfaces,
            surface_mode_contract=contract,
        )

        self.assertEqual(finite_current_mode, module.DEFAULT_FINITE_CURRENT_MODE)
        self.assertEqual(settings["mode"], "vacuum")
        self.assertEqual(settings["effective_mode"], "vacuum")
        self.assertEqual(settings["plasma_current_A"], 0.0)

        inherited_nonvacuum_mode = (
            module.resolve_stage2_finite_current_mode_for_surface_mode(
                {"FINITE_CURRENT_MODE": "jhalpern30_proxy_field"},
                requested_args.finite_current_mode,
                contract,
            )
        )
        with self.assertRaisesRegex(ValueError, "vacuum-locked"):
            module.resolve_plasma_current_settings(
                requested_args,
                finite_current_mode=inherited_nonvacuum_mode,
                default_plasma_current_A=0.0,
                num_surfaces=contract.num_surfaces,
                surface_mode_contract=contract,
            )

    def test_published_stage2_seed_requires_solved_iota_and_G(self):
        module = self.load_module()
        contract = module.resolve_surface_mode_contract(
            SimpleNamespace(
                surface_mode=module.PUBLISHED_MULTISURFACE,
                num_surfaces=1,
                inner_surface_ratio=0.8,
            ),
            warn_on_legacy_mapping=False,
        )

        with patch.object(
            module,
            "initialize_boozer_surface",
            side_effect=AssertionError(
                "surface-only seed must fail before Boozer solve"
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "explicit iota and G"):
                module.initialize_surface_data_for_contract(
                    [
                        {
                            "name": "inner0",
                            "seed_label": 0.15,
                            "target_volume": 0.06,
                            "initial_surface": object(),
                        },
                        {
                            "name": "inner1",
                            "seed_label": 0.20,
                            "target_volume": 0.08,
                            "initial_surface": object(),
                        },
                        {
                            "name": "outer",
                            "seed_label": 0.25,
                            "target_volume": 0.10,
                            "initial_surface": object(),
                        },
                    ],
                    surface_mode_contract=contract,
                    mpol=8,
                    ntor=6,
                    bs=object(),
                    constraint_weight=1.0,
                    default_iota=0.15,
                    default_G=1.0,
                    boozer_I=0.0,
                    nfp=5,
                    stage2_seed_surface=SimpleNamespace(
                        surface=SimpleNamespace(name="stage2_outer"),
                        iota=0.21,
                        G=None,
                    ),
                    warm_start_surface_stem=None,
                )

    def test_published_stage2_seed_completion_uses_bootability_iota_and_signed_G(self):
        module = self.load_module()
        seed = module.WarmStartBoozerSeed(
            surface=SimpleNamespace(name="stage2_outer"),
            iota=None,
            G=None,
            source_path=Path("/tmp/stage2/surf_opt_boozer_surface.json"),
        )

        completed = module.complete_published_stage2_seed_surface(
            seed,
            {"BOOTABILITY_SOLVED_IOTA": 0.1476},
            initial_G=-2.0106,
        )

        self.assertIs(completed.surface, seed.surface)
        self.assertEqual(completed.source_path, seed.source_path)
        self.assertAlmostEqual(completed.iota, 0.1476)
        self.assertAlmostEqual(completed.G, -2.0106)

    def test_published_stage2_seed_completion_requires_bootability_iota(self):
        module = self.load_module()
        seed = module.WarmStartBoozerSeed(
            surface=SimpleNamespace(name="stage2_outer"),
            iota=None,
            G=-2.0106,
            source_path=Path("/tmp/stage2/surf_opt_boozer_surface.json"),
        )

        with self.assertRaisesRegex(RuntimeError, "BOOTABILITY_SOLVED_IOTA"):
            module.complete_published_stage2_seed_surface(
                seed,
                {},
                initial_G=-2.0106,
            )

    def test_published_stage2_seed_rejects_solved_G_drift(self):
        module = self.load_module()
        family = importlib.import_module("banana_opt.boozer_surface_family")
        contract = module.resolve_surface_mode_contract(
            SimpleNamespace(
                surface_mode=module.PUBLISHED_MULTISURFACE,
                num_surfaces=1,
                inner_surface_ratio=0.8,
            ),
            warn_on_legacy_mapping=False,
        )

        class FakeSurface:
            def __init__(self, name, volume):
                self.name = name
                self._volume = volume

            def volume(self):
                return self._volume

        def fake_initialize_boozer_surface(
            initial_surface,
            _mpol,
            _ntor,
            _bs,
            _vol_target,
            _constraint_weight,
            iota,
            G0,
            _boozer_I,
            *,
            initial_surface_guess,
            nfp,
        ):
            del iota, initial_surface_guess, nfp
            solved_G = (
                0.77 if initial_surface.name == "stage2_outer" else float(G0) + 0.10
            )
            solved_volume = {
                "stage2_outer": 0.10,
                "contracted_solved_stage2_outer_to_0.08": 0.08,
                "contracted_solved_contracted_solved_stage2_outer_to_0.08_to_0.06": 0.06,
            }[initial_surface.name]
            return SimpleNamespace(
                success=True,
                solve_success=True,
                self_intersecting=False,
                solved_iota=0.15,
                solved_G=solved_G,
                volume=solved_volume,
                boozer_surface=SimpleNamespace(
                    surface=FakeSurface(
                        f"solved_{initial_surface.name}", solved_volume
                    ),
                    res={"iota": 0.15, "G": solved_G},
                ),
            )

        def fake_contract_surface_to_target_volume(previous_surface, target_volume):
            return FakeSurface(
                f"contracted_{previous_surface.name}_to_{target_volume:.2f}",
                target_volume,
            )

        with (
            patch.object(
                family,
                "attempt_initialize_boozer_surface",
                side_effect=fake_initialize_boozer_surface,
            ),
            patch.object(
                module,
                "contract_surface_to_target_volume",
                side_effect=fake_contract_surface_to_target_volume,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "inner0"):
                module.initialize_surface_data_for_contract(
                    [
                        {
                            "name": "inner0",
                            "seed_label": 0.15,
                            "target_volume": 0.06,
                            "initial_surface": object(),
                        },
                        {
                            "name": "inner1",
                            "seed_label": 0.20,
                            "target_volume": 0.08,
                            "initial_surface": object(),
                        },
                        {
                            "name": "outer",
                            "seed_label": 0.25,
                            "target_volume": 0.10,
                            "initial_surface": object(),
                        },
                    ],
                    surface_mode_contract=contract,
                    mpol=8,
                    ntor=6,
                    bs=object(),
                    constraint_weight=1.0,
                    default_iota=0.15,
                    default_G=1.0,
                    boozer_I=0.0,
                    nfp=5,
                    stage2_seed_surface=SimpleNamespace(
                        surface=SimpleNamespace(name="stage2_outer"),
                        iota=0.21,
                        G=0.77,
                    ),
                    warm_start_surface_stem=None,
                )

    def test_published_stage2_seed_rejects_nonmonotone_actual_volumes(self):
        module = self.load_module()
        family = importlib.import_module("banana_opt.boozer_surface_family")
        contract = module.resolve_surface_mode_contract(
            SimpleNamespace(
                surface_mode=module.PUBLISHED_MULTISURFACE,
                num_surfaces=1,
                inner_surface_ratio=0.8,
            ),
            warn_on_legacy_mapping=False,
        )

        class FakeSurface:
            def __init__(self, name, volume):
                self.name = name
                self._volume = volume

            def volume(self):
                return self._volume

        def fake_initialize_boozer_surface(
            initial_surface,
            _mpol,
            _ntor,
            _bs,
            _vol_target,
            _constraint_weight,
            iota,
            G0,
            _boozer_I,
            *,
            initial_surface_guess,
            nfp,
        ):
            del initial_surface_guess, nfp
            solved_name = {
                "stage2_outer": "solved_outer",
                "contracted_solved_outer_to_0.08": "solved_inner1",
                "contracted_solved_inner1_to_0.06": "solved_inner0",
            }[initial_surface.name]
            solved_volume = {
                "solved_outer": 0.10,
                "solved_inner1": 0.11,
                "solved_inner0": 0.12,
            }[solved_name]
            return SimpleNamespace(
                success=True,
                solve_success=True,
                self_intersecting=False,
                solved_iota=float(iota),
                solved_G=float(G0),
                volume=solved_volume,
                boozer_surface=SimpleNamespace(
                    surface=FakeSurface(solved_name, solved_volume),
                    res={"iota": float(iota), "G": float(G0)},
                ),
            )

        def fake_contract_surface_to_target_volume(previous_surface, target_volume):
            return FakeSurface(
                f"contracted_{previous_surface.name}_to_{target_volume:.2f}",
                target_volume,
            )

        with (
            patch.object(
                family,
                "attempt_initialize_boozer_surface",
                side_effect=fake_initialize_boozer_surface,
            ),
            patch.object(
                module,
                "contract_surface_to_target_volume",
                side_effect=fake_contract_surface_to_target_volume,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "solved volumes"):
                module.initialize_surface_data_for_contract(
                    [
                        {
                            "name": "inner0",
                            "seed_label": 0.15,
                            "target_volume": 0.06,
                            "initial_surface": object(),
                        },
                        {
                            "name": "inner1",
                            "seed_label": 0.20,
                            "target_volume": 0.08,
                            "initial_surface": object(),
                        },
                        {
                            "name": "outer",
                            "seed_label": 0.25,
                            "target_volume": 0.10,
                            "initial_surface": object(),
                        },
                    ],
                    surface_mode_contract=contract,
                    mpol=8,
                    ntor=6,
                    bs=object(),
                    constraint_weight=1.0,
                    default_iota=0.15,
                    default_G=1.0,
                    boozer_I=0.0,
                    nfp=5,
                    stage2_seed_surface=SimpleNamespace(
                        surface=SimpleNamespace(name="stage2_outer"),
                        iota=0.21,
                        G=0.77,
                    ),
                    warm_start_surface_stem=None,
                )

    def test_published_stage2_seed_rejects_non_positive_actual_volumes(self):
        module = self.load_module()

        def surface(name, volume):
            return SimpleNamespace(name=name, volume=lambda: volume)

        surface_data = [
            {
                "name": "inner0",
                "boozer_surface": SimpleNamespace(surface=surface("inner0", -0.12)),
            },
            {
                "name": "inner1",
                "boozer_surface": SimpleNamespace(surface=surface("inner1", -0.11)),
            },
            {
                "name": "outer",
                "boozer_surface": SimpleNamespace(surface=surface("outer", -0.10)),
            },
        ]

        with self.assertRaisesRegex(RuntimeError, "finite and positive"):
            module._require_published_volume_order(surface_data)

    def test_contract_surface_to_target_volume_rejects_non_positive_volumes(self):
        module = self.load_module()
        positive_neighbor = SimpleNamespace(volume=lambda: 0.10)
        negative_neighbor = SimpleNamespace(volume=lambda: -0.10)

        with self.assertRaisesRegex(RuntimeError, "Continuation solved neighbor"):
            module.contract_surface_to_target_volume(negative_neighbor, 0.08)
        with self.assertRaisesRegex(RuntimeError, "Continuation target"):
            module.contract_surface_to_target_volume(positive_neighbor, 0.0)

    def test_published_without_seed_or_warm_start_rejects_cold_wout_inner_init(self):
        module = self.load_module()
        contract = module.resolve_surface_mode_contract(
            SimpleNamespace(
                surface_mode=module.PUBLISHED_MULTISURFACE,
                num_surfaces=1,
                inner_surface_ratio=0.8,
            ),
            warn_on_legacy_mapping=False,
        )

        with self.assertRaisesRegex(RuntimeError, "cold WOUT inner initialization"):
            module.initialize_surface_data_for_contract(
                [
                    {
                        "name": "inner0",
                        "seed_label": 0.15,
                        "target_volume": 0.06,
                        "initial_surface": object(),
                    },
                    {
                        "name": "inner1",
                        "seed_label": 0.20,
                        "target_volume": 0.08,
                        "initial_surface": object(),
                    },
                    {
                        "name": "outer",
                        "seed_label": 0.25,
                        "target_volume": 0.10,
                        "initial_surface": object(),
                    },
                ],
                surface_mode_contract=contract,
                mpol=8,
                ntor=6,
                bs=object(),
                constraint_weight=1.0,
                default_iota=0.15,
                default_G=1.0,
                boozer_I=0.0,
                nfp=5,
                stage2_seed_surface=None,
                warm_start_surface_stem=None,
            )

    def test_published_named_warm_start_stack_remains_strict(self):
        module = self.load_module()
        contract = module.resolve_surface_mode_contract(
            SimpleNamespace(
                surface_mode=module.PUBLISHED_MULTISURFACE,
                num_surfaces=1,
                inner_surface_ratio=0.8,
            ),
            warn_on_legacy_mapping=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            warm_start_stem = str(Path(tmpdir) / "pubms_seed")
            with patch.object(
                module,
                "initialize_boozer_surface",
                side_effect=AssertionError(
                    "missing named warm-start stack must fail before Boozer solve"
                ),
            ):
                with self.assertRaises(FileNotFoundError):
                    module.initialize_surface_data_for_contract(
                        [
                            {
                                "name": "inner0",
                                "seed_label": 0.15,
                                "target_volume": 0.06,
                                "initial_surface": object(),
                            },
                            {
                                "name": "inner1",
                                "seed_label": 0.20,
                                "target_volume": 0.08,
                                "initial_surface": object(),
                            },
                            {
                                "name": "outer",
                                "seed_label": 0.25,
                                "target_volume": 0.10,
                                "initial_surface": object(),
                            },
                        ],
                        surface_mode_contract=contract,
                        mpol=8,
                        ntor=6,
                        bs=object(),
                        constraint_weight=1.0,
                        default_iota=0.15,
                        default_G=1.0,
                        boozer_I=0.0,
                        nfp=5,
                        stage2_seed_surface=SimpleNamespace(
                            surface=SimpleNamespace(name="stage2_outer"),
                            iota=0.21,
                            G=0.77,
                        ),
                        warm_start_surface_stem=warm_start_stem,
                    )

    def test_published_named_warm_start_stack_loads_all_artifacts_before_solving(self):
        module = self.load_module()
        contract = module.resolve_surface_mode_contract(
            SimpleNamespace(
                surface_mode=module.PUBLISHED_MULTISURFACE,
                num_surfaces=1,
                inner_surface_ratio=0.8,
            ),
            warn_on_legacy_mapping=False,
        )
        loaded_paths = []

        def fake_warm_start_path(_stem, *, surface_name):
            return Path(f"/tmp/pubms_seed_{surface_name}_boozer_surface.json")

        def fake_load_warm_start_boozer_seed(path):
            loaded_paths.append(path.name)
            if "inner0" not in path.name:
                raise FileNotFoundError(path)
            return module.WarmStartBoozerSeed(
                surface=SimpleNamespace(name="warm_inner0"),
                iota=0.21,
                G=0.77,
                source_path=path,
            )

        with (
            patch.object(
                module,
                "resolve_warm_start_boozer_surface_path",
                side_effect=fake_warm_start_path,
            ),
            patch.object(
                module,
                "load_warm_start_boozer_seed",
                side_effect=fake_load_warm_start_boozer_seed,
            ),
            patch.object(
                module,
                "initialize_boozer_surface",
                side_effect=AssertionError(
                    "partial warm-start stack must fail before Boozer solve"
                ),
            ) as initialize_mock,
        ):
            with self.assertRaises(FileNotFoundError):
                module.initialize_surface_data_for_contract(
                    [
                        {
                            "name": "inner0",
                            "seed_label": 0.15,
                            "target_volume": 0.06,
                            "initial_surface": object(),
                        },
                        {
                            "name": "inner1",
                            "seed_label": 0.20,
                            "target_volume": 0.08,
                            "initial_surface": object(),
                        },
                        {
                            "name": "outer",
                            "seed_label": 0.25,
                            "target_volume": 0.10,
                            "initial_surface": object(),
                        },
                    ],
                    surface_mode_contract=contract,
                    mpol=8,
                    ntor=6,
                    bs=object(),
                    constraint_weight=1.0,
                    default_iota=0.15,
                    default_G=1.0,
                    boozer_I=0.0,
                    nfp=5,
                    stage2_seed_surface=SimpleNamespace(
                        surface=SimpleNamespace(name="stage2_outer"),
                        iota=0.21,
                        G=0.77,
                    ),
                    warm_start_surface_stem="/tmp/pubms_seed",
                )

        self.assertEqual(
            loaded_paths,
            [
                "pubms_seed_inner0_boozer_surface.json",
                "pubms_seed_inner1_boozer_surface.json",
            ],
        )
        initialize_mock.assert_not_called()

    def test_published_named_warm_start_stack_enforces_published_postconditions(self):
        module = self.load_module()
        contract = module.resolve_surface_mode_contract(
            SimpleNamespace(
                surface_mode=module.PUBLISHED_MULTISURFACE,
                num_surfaces=1,
                inner_surface_ratio=0.8,
            ),
            warn_on_legacy_mapping=False,
        )
        surface_configs = [
            {
                "name": "inner0",
                "seed_label": 0.15,
                "target_volume": 0.06,
                "initial_surface": SimpleNamespace(name="wout_inner0"),
            },
            {
                "name": "inner1",
                "seed_label": 0.20,
                "target_volume": 0.08,
                "initial_surface": SimpleNamespace(name="wout_inner1"),
            },
            {
                "name": "outer",
                "seed_label": 0.25,
                "target_volume": 0.10,
                "initial_surface": SimpleNamespace(name="wout_outer"),
            },
        ]

        class FakeSurface:
            def __init__(self, name, volume):
                self.name = name
                self._volume = volume

            def volume(self):
                return self._volume

        def run_warm_started_publish_stack(*, solved_volumes, solved_G):
            def fake_warm_start_path(_stem, *, surface_name):
                return Path(f"/tmp/pubms_seed_{surface_name}_boozer_surface.json")

            def fake_load_warm_start_boozer_seed(path):
                surface_name = path.name.removeprefix("pubms_seed_").removesuffix(
                    "_boozer_surface.json"
                )
                return module.WarmStartBoozerSeed(
                    surface=SimpleNamespace(name=f"warm_{surface_name}"),
                    iota=0.21,
                    G=0.77,
                    source_path=path,
                )

            def fake_initialize_boozer_surface(
                initial_surface,
                _mpol,
                _ntor,
                _bs,
                _vol_target,
                _constraint_weight,
                iota,
                _G0,
                _boozer_I,
                *,
                initial_surface_guess,
                nfp,
            ):
                del initial_surface_guess, nfp
                surface_name = initial_surface.name.removeprefix("warm_")
                return SimpleNamespace(
                    surface=FakeSurface(
                        f"solved_{surface_name}", solved_volumes[surface_name]
                    ),
                    res={"iota": float(iota), "G": solved_G[surface_name]},
                )

            with (
                patch.object(
                    module,
                    "resolve_warm_start_boozer_surface_path",
                    side_effect=fake_warm_start_path,
                ),
                patch.object(
                    module,
                    "load_warm_start_boozer_seed",
                    side_effect=fake_load_warm_start_boozer_seed,
                ),
                patch.object(
                    module,
                    "initialize_boozer_surface",
                    side_effect=fake_initialize_boozer_surface,
                ),
                patch.object(
                    module,
                    "cross_sections_are_nested",
                    return_value=(True, []),
                ),
            ):
                return module.initialize_surface_data_for_contract(
                    surface_configs,
                    surface_mode_contract=contract,
                    mpol=8,
                    ntor=6,
                    bs=object(),
                    constraint_weight=1.0,
                    default_iota=0.15,
                    default_G=1.0,
                    boozer_I=0.0,
                    nfp=5,
                    stage2_seed_surface=SimpleNamespace(
                        surface=SimpleNamespace(name="stage2_outer"),
                        iota=0.21,
                        G=0.77,
                    ),
                    warm_start_surface_stem="/tmp/pubms_seed",
                )

        with self.subTest("G drift"):
            with self.assertRaisesRegex(RuntimeError, "solved G drifted"):
                run_warm_started_publish_stack(
                    solved_volumes={"inner0": 0.06, "inner1": 0.08, "outer": 0.10},
                    solved_G={"inner0": 0.77, "inner1": 0.88, "outer": 0.77},
                )

        with self.subTest("volume order"):
            with self.assertRaisesRegex(RuntimeError, "solved volumes"):
                run_warm_started_publish_stack(
                    solved_volumes={"inner0": 0.06, "inner1": 0.11, "outer": 0.10},
                    solved_G={"inner0": 0.77, "inner1": 0.77, "outer": 0.77},
                )

    def test_single_surface_initialization_keeps_config_order_without_continuation(
        self,
    ):
        module = self.load_module()
        contract = module.resolve_surface_mode_contract(
            SimpleNamespace(
                surface_mode=module.SINGLE_SURFACE,
                num_surfaces=1,
                inner_surface_ratio=0.8,
            ),
            warn_on_legacy_mapping=False,
        )
        init_sources = []

        def fake_initialize_boozer_surface(
            initial_surface,
            *_args,
            initial_surface_guess,
            **_kwargs,
        ):
            init_sources.append((initial_surface.name, initial_surface_guess))
            return SimpleNamespace(
                surface=SimpleNamespace(name=f"solved_{initial_surface.name}"),
                res={"iota": 0.15, "G": 1.0},
            )

        with (
            patch.object(
                module,
                "initialize_boozer_surface",
                side_effect=fake_initialize_boozer_surface,
            ),
            patch.object(
                module,
                "contract_surface_to_target_volume",
                side_effect=AssertionError("single_surface must not use continuation"),
            ),
        ):
            surface_data, _warm_paths = module.initialize_surface_data_for_contract(
                [
                    {
                        "name": "outer",
                        "seed_label": 0.25,
                        "target_volume": 0.10,
                        "initial_surface": SimpleNamespace(name="wout_outer"),
                    }
                ],
                surface_mode_contract=contract,
                mpol=8,
                ntor=6,
                bs=object(),
                constraint_weight=1.0,
                default_iota=0.15,
                default_G=1.0,
                boozer_I=0.0,
                nfp=5,
                stage2_seed_surface=None,
                warm_start_surface_stem=None,
            )

        self.assertEqual(init_sources, [("wout_outer", None)])
        self.assertEqual([entry["name"] for entry in surface_data], ["outer"])

    def test_average_surface_objectives_uses_mean_and_preserves_single_surface_scale(
        self,
    ):
        module = self.load_module()

        single = FakeAlgebraicObjective(2.0, [2.0, -1.0])
        single_avg = module.average_surface_objectives([single])
        self.assertAlmostEqual(single_avg.J(), 2.0)
        np.testing.assert_allclose(single_avg.dJ(), [2.0, -1.0])

        left = FakeAlgebraicObjective(2.0, [2.0, -1.0])
        right = FakeAlgebraicObjective(6.0, [4.0, 3.0])
        pair_avg = module.average_surface_objectives([left, right])
        self.assertAlmostEqual(pair_avg.J(), 4.0)
        np.testing.assert_allclose(pair_avg.dJ(), [3.0, 1.0])

    def test_build_surface_search_weights_ramps_inner_surface_only(self):
        module = self.load_module()

        np.testing.assert_allclose(
            module.build_surface_search_weights(
                num_surfaces=1,
                accepted_iterations=0,
                ramp_iterations=5,
                initial_inner_weight=0.0,
            ),
            [1.0],
        )
        np.testing.assert_allclose(
            module.build_surface_search_weights(
                num_surfaces=2,
                accepted_iterations=0,
                ramp_iterations=5,
                initial_inner_weight=0.0,
            ),
            [0.0, 1.0],
        )
        np.testing.assert_allclose(
            module.build_surface_search_weights(
                num_surfaces=2,
                accepted_iterations=2,
                ramp_iterations=5,
                initial_inner_weight=0.0,
            ),
            [0.4, 1.0],
        )
        np.testing.assert_allclose(
            module.build_surface_search_weights(
                num_surfaces=2,
                accepted_iterations=5,
                ramp_iterations=5,
                initial_inner_weight=0.0,
            ),
            [1.0, 1.0],
        )

    def test_build_surface_search_gate_ramps_thresholds_and_nesting(self):
        module = self.load_module()

        single = module.build_surface_search_gate(
            num_surfaces=1,
            accepted_iterations=0,
            ramp_iterations=5,
            initial_inner_weight=0.0,
            surface_gap_threshold=0.005,
        )
        self.assertEqual(single["surface_gap_threshold"], 0.005)
        self.assertTrue(single["enforce_nesting"])
        self.assertEqual(single["gate_scale"], 1.0)

        start = module.build_surface_search_gate(
            num_surfaces=2,
            accepted_iterations=0,
            ramp_iterations=5,
            initial_inner_weight=0.0,
            surface_gap_threshold=0.005,
        )
        self.assertEqual(start["surface_gap_threshold"], 0.0)
        self.assertFalse(start["enforce_nesting"])
        self.assertEqual(start["gate_scale"], 0.0)

        mid = module.build_surface_search_gate(
            num_surfaces=2,
            accepted_iterations=2,
            ramp_iterations=5,
            initial_inner_weight=0.0,
            surface_gap_threshold=0.005,
        )
        self.assertAlmostEqual(mid["surface_gap_threshold"], 0.002)
        self.assertFalse(mid["enforce_nesting"])
        self.assertAlmostEqual(mid["gate_scale"], 0.4)

        done = module.build_surface_search_gate(
            num_surfaces=2,
            accepted_iterations=5,
            ramp_iterations=5,
            initial_inner_weight=0.0,
            surface_gap_threshold=0.005,
        )
        self.assertAlmostEqual(done["surface_gap_threshold"], 0.005)
        self.assertTrue(done["enforce_nesting"])
        self.assertAlmostEqual(done["gate_scale"], 1.0)

    def test_published_surface_search_policy_uses_fixed_weights_and_strict_gate(self):
        module = self.load_module()
        contract = module.resolve_surface_mode_contract(
            SimpleNamespace(
                surface_mode=module.PUBLISHED_MULTISURFACE,
                num_surfaces=1,
                inner_surface_ratio=0.8,
            ),
            warn_on_legacy_mapping=False,
        )

        weights = module.build_surface_search_weights_for_contract(
            contract,
            accepted_iterations=0,
            ramp_iterations=5,
            initial_inner_weight=0.0,
        )
        gate = module.build_surface_search_gate_for_contract(
            contract,
            accepted_iterations=0,
            ramp_iterations=5,
            initial_inner_weight=0.0,
            surface_gap_threshold=0.005,
        )

        np.testing.assert_allclose(weights, [1.0, 1.0, 1.0])
        self.assertEqual(gate["surface_gap_threshold"], 0.005)
        self.assertTrue(gate["enforce_nesting"])
        self.assertEqual(gate["gate_scale"], 1.0)


class HardwareConstraintTests(unittest.TestCase):
    def load_module(self):
        return load_single_stage_example_module()

    @staticmethod
    def inboard_midplane_curve():
        return SimpleNamespace(gamma=lambda: np.array([[0.876, 0.0, 0.0]], dtype=float))

    def _run_fun_with_hardware_violation(
        self,
        *,
        hardware_search_mode="hard",
        hardware_search_soft_iterations=0,
        accepted_iterations=0,
        curvature_traversal_band=0.0,
        curvature_traversal_eval_budget=0,
        search_hardware_values=(0.25, 0.0, 0.0),
    ):
        module = load_single_stage_example_module()

        last_J = 12.0
        last_dJ = np.array([1.0, -1.0, 2.0])

        class _Surface:
            x = np.ones(2)

            def is_self_intersecting(self):
                return False

            def volume(self):
                return 1.0

            def gamma(self):
                return np.zeros((1, 1, 3))

        class _BoozerSurface:
            surface = _Surface()
            res = {"success": True, "iota": TEST_IOTA, "G": TEST_G0}

            def run_code(self, iota, G):
                return self.res

        class _JF:
            x = np.zeros(3)

        class _DistanceObjective:
            def __init__(self, distance):
                self.distance = distance

            def shortest_distance(self):
                return self.distance

        class _CurvatureObjective:
            def J(self):
                return 1.0

            def dJ(self):
                return np.zeros(3)

        class _LengthObjective:
            def J(self):
                return 1.0

            def dJ(self):
                return np.zeros(3)

        class _Curve:
            def kappa(self):
                return np.array([41.0])

            def gamma(self):
                return np.zeros((2, 3))

        surface_data = [
            {"boozer_surface": _BoozerSurface()},
            {"boozer_surface": _BoozerSurface()},
        ]
        module.run_dict = {
            "x_prev": np.zeros(3),
            "lscount": 0,
            "surface_state": {
                "sdofs": [np.ones(2), np.ones(2)],
                "iota": [TEST_IOTA, TEST_IOTA],
                "G": [TEST_G0, TEST_G0],
            },
            "J": last_J,
            "dJ": last_dJ.copy(),
            "accepted_iterations": accepted_iterations,
            "accepted_x": np.zeros(3),
            "trial_hardware_status": None,
            "accepted_hardware_status": None,
        }
        module.surface_data = surface_data
        module.outer_surface_data = surface_data[-1]
        module.surface_iota_terms = [
            SimpleNamespace(J=lambda: TEST_IOTA),
            SimpleNamespace(J=lambda: TEST_IOTA),
        ]
        module.VV = object()
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.JF = _JF()
        module.bs = object()
        module.nonQSs = []
        module.brs = []
        module.Jiota = object()
        module.IOTAS_WEIGHT = 1.0
        module.JCurveLength = _LengthObjective()
        seed_near_miss_diagnostic_globals(module, coil_length=module.JCurveLength.J())
        module.LENGTH_WEIGHT = 1.0
        module.JCurveCurve = _DistanceObjective(0.04)
        module.CC_WEIGHT = 1.0
        module.CC_DIST = 0.05
        module.JCurveSurface = _DistanceObjective(0.03)
        module.CS_WEIGHT = 1.0
        module.CS_DIST = 0.02
        module.JCurvature = _CurvatureObjective()
        module.CURVATURE_WEIGHT = 1.0
        module.CURVATURE_THRESHOLD = 40.0
        module.RES_WEIGHT = 1.0
        module.MULTISURFACE_RAMP_ITERATIONS = 0
        module.INNER_SURFACE_INITIAL_WEIGHT = 1.0
        module.TOPOLOGY_GATE_FIELDLINES = 0
        module.TOPOLOGY_GATE_TMAX = 2.0
        module.TOPOLOGY_GATE_TOL = 1e-7
        module.TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.25
        module.TOPOLOGY_GATE_PENALTY_SCALE = 4.0
        module.HARDWARE_SEARCH_MODE = hardware_search_mode
        module.HARDWARE_SEARCH_SOFT_ITERATIONS = hardware_search_soft_iterations
        module.CURVATURE_TRAVERSAL_BAND = curvature_traversal_band
        module.CURVATURE_TRAVERSAL_EVAL_BUDGET = curvature_traversal_eval_budget
        module.banana_curve = _Curve()

        with (
            patch.object(
                module,
                "solve_surface_stack_at_dofs",
                return_value={
                    "success": True,
                    "solve_success": [True, True],
                    "self_intersections": [False, False],
                    "volumes_ordered": True,
                    "gap_ok": True,
                    "nesting_ok": True,
                    "adjacent_gaps": [0.1],
                    "outer_vessel_gap": 0.05,
                    "bad_nesting_phis": [],
                },
            ),
            patch.object(
                module,
                "evaluate_total_objective",
                return_value={
                    "total": 7.0,
                    "grad": np.arange(3, dtype=float),
                    "surface_weights": np.array([1.0, 1.0]),
                    "J_QS": 0.0,
                    "dJ_QS": np.zeros(3),
                    "J_Boozer": 0.0,
                    "dJ_Boozer": np.zeros(3),
                    "J_iota": 0.0,
                    "dJ_iota": np.zeros(3),
                    "J_curvature": 0.0,
                    "dJ_curvature": np.zeros(3),
                    **search_hardware_penalty_payload(search_hardware_values),
                },
            ),
            patch.object(module, "restore_surface_states") as restore_mock,
            patch.object(
                module,
                "evaluate_search_topology_gate",
                return_value={
                    "enabled": False,
                    "success": True,
                    "nfieldlines": 0,
                    "survived_lines": 0,
                    "survival_fraction": 1.0,
                    "survival_threshold": 0.25,
                    "tmax": 2.0,
                    "tol": 1e-7,
                    "stop_reason_counts": {},
                    "first_exit_time": None,
                    "first_exit_angle": None,
                    "first_exit_reason": None,
                },
            ),
            patch.object(
                module,
                "evaluate_surface_stack",
                return_value={
                    "success": True,
                    "solve_success": [True, True],
                    "self_intersections": [False, False],
                    "volumes_ordered": True,
                    "gap_ok": True,
                    "nesting_ok": True,
                    "adjacent_gaps": [0.1],
                    "outer_vessel_gap": 0.05,
                    "bad_nesting_phis": [],
                },
            ),
            patch.object(
                module,
                "evaluate_single_stage_hardware_snapshot",
                return_value=topology_hardware_snapshot(),
            ),
            patch.object(
                module,
                "compute_surface_field_metrics",
                return_value=(0.0, 0.0),
            ),
            patch.object(
                module,
                "maybe_write_best_hardware_near_miss_trial_artifacts",
                return_value=False,
            ),
        ):
            J_out, dJ_out = module.fun(np.ones(3))

        return module, J_out, dJ_out, last_dJ, restore_mock

    def test_stage2_hardware_constraints_report_each_violation(self):
        module = load_stage2_module()

        status = module.evaluate_stage2_hardware_constraints(
            coil_length=2.1,
            length_target=1.75,
            curve_curve_min_dist=0.04,
            cc_threshold=0.05,
            max_curvature=41.0,
            curvature_threshold=40.0,
        )

        self.assertFalse(status["success"])
        self.assertEqual(len(status["violations"]), 3)
        self.assertIn("coil_coil_spacing", status["violations"][0])
        self.assertIn("max_curvature", status["violations"][1])
        self.assertIn("coil_length", status["violations"][2])

    def test_stage2_hardware_constraints_enforce_length_floor(self):
        module = load_stage2_module()

        status = module.evaluate_stage2_hardware_constraints(
            coil_length=0.8,
            length_target=1.9,
            curve_curve_min_dist=0.05,
            cc_threshold=0.05,
            max_curvature=40.0,
            curvature_threshold=40.0,
        )

        self.assertFalse(status["success"])
        self.assertEqual(
            status["violations"],
            ["coil_length_min 0.800000 below threshold 0.950000"],
        )

    def test_stage2_hardware_constraints_use_hard_upper_length_for_artifacts(self):
        module = load_stage2_module()

        status = module.evaluate_stage2_hardware_constraints(
            coil_length=1.95,
            length_target=1.9,
            curve_curve_min_dist=0.05,
            cc_threshold=0.05,
            max_curvature=40.0,
            curvature_threshold=40.0,
        )

        self.assertTrue(status["success"])
        self.assertEqual(
            status["constraints"]["coil_length"]["threshold"],
            module.COIL_LENGTH_HARD_LIMIT_M,
        )
        self.assertEqual(status["length_target"], 1.9)

    def test_stage2_hardware_constraints_enforce_self_envelope_and_fold(self):
        module = load_stage2_module()

        status = module.evaluate_stage2_hardware_constraints(
            coil_length=1.95,
            length_target=1.9,
            curve_curve_min_dist=0.05,
            cc_threshold=0.05,
            max_curvature=40.0,
            curvature_threshold=40.0,
            self_envelope_min_dist=0.0460,
            self_envelope_min_distance=0.0462,
            fold_geodesic_curvature_max=44.0,
            fold_geodesic_curvature_limit=43.0,
        )

        self.assertFalse(status["success"])
        self.assertEqual(
            status["violations"],
            [
                "self_envelope_min_dist 0.046000 below threshold 0.046200",
                "fold_geodesic_curvature_max 44.000000 exceeds threshold 43.000000",
            ],
        )
        self.assertFalse(status["constraints"]["self_envelope_min_dist"]["success"])
        self.assertFalse(
            status["constraints"]["fold_geodesic_curvature_max"]["success"]
        )

    def test_single_stage_hardware_constraints_report_each_violation(self):
        module = load_single_stage_example_module()

        status = module.evaluate_single_stage_hardware_constraints(
            curve_curve_min_dist=0.04,
            cc_dist=0.05,
            curve_surface_min_dist=0.01,
            cs_dist=0.02,
            surface_vessel_min_dist=0.05,
            ss_dist=0.04,
            max_curvature=41.0,
            curvature_threshold=40.0,
        )

        self.assertFalse(status["success"])
        self.assertEqual(len(status["violations"]), 3)
        self.assertIn("coil_coil_spacing", status["violations"][0])
        self.assertIn("coil_surface_spacing", status["violations"][1])
        self.assertIn("max_curvature", status["violations"][2])

    def test_single_stage_hardware_constraints_report_current_and_length_violations(
        self,
    ):
        module = load_single_stage_example_module()

        status = module.evaluate_single_stage_hardware_constraints(
            curve_curve_min_dist=0.05,
            cc_dist=0.05,
            curve_surface_min_dist=0.02,
            cs_dist=0.02,
            surface_vessel_min_dist=0.05,
            ss_dist=0.04,
            max_curvature=40.0,
            curvature_threshold=40.0,
            coil_length=2.1,
            length_target=1.7,
            poloidal_extent_rad=0.7,
            poloidal_extent_threshold_rad=0.8,
            tf_current_A=9.0e4,
            tf_current_limit_A=8.0e4,
            banana_current_A=1.7e4,
            banana_current_max_A=1.6e4,
            coil_width=0.12,
            width_min_threshold=0.05,
            width_max_threshold=0.17,
            self_intersect_penalty=0.0,
            self_intersect_threshold=0.0,
            lcfs_major_radius_m=in_bounds_lcfs_major_radius_m(),
            lcfs_minor_radius_m=in_bounds_lcfs_minor_radius_m(),
        )
        search_status = status["search_hardware_status"]
        artifact_status = status["artifact_hardware_status"]

        self.assertFalse(search_status["success"])
        self.assertEqual(
            search_status["violations"],
            ["|banana_current| 17000.000000 exceeds threshold 16000.000000"],
        )
        self.assertEqual(
            search_status["allowed_traversal_status"]["violations"],
            [],
        )
        self.assertEqual(
            search_status["forbidden_traversal_status"]["violations"],
            ["|banana_current| 17000.000000 exceeds threshold 16000.000000"],
        )
        self.assertFalse(artifact_status["success"])
        self.assertEqual(len(artifact_status["violations"]), 3)
        self.assertEqual(
            status["curve_curve_distance_metric_kind"],
            stage2_solver.POINT_CLOUD_MINIMUM_CAPPED_AT_THRESHOLD_METRIC_KIND,
        )
        self.assertEqual(
            status["curve_surface_distance_metric_kind"],
            stage2_solver.POINT_CLOUD_MINIMUM_CAPPED_AT_THRESHOLD_METRIC_KIND,
        )
        self.assertIn("coil_length", artifact_status["violations"][0])
        self.assertIn("banana_current", artifact_status["violations"][1])
        self.assertIn("tf_current", artifact_status["violations"][2])
        self.assertEqual(
            artifact_status["allowed_traversal_status"]["violations"],
            ["coil_length 2.100000 exceeds threshold 2.000000"],
        )
        self.assertEqual(
            artifact_status["forbidden_traversal_status"]["violations"],
            [
                "|banana_current| 17000.000000 exceeds threshold 16000.000000",
                "|tf_current| 90000.000000 exceeds threshold 80000.000000",
            ],
        )
        self.assertEqual(
            artifact_status["constraints"]["coil_length"]["threshold"],
            2.0,
        )
        self.assertEqual(
            artifact_status["constraints"]["tf_current"]["threshold"],
            8.0e4,
        )
        self.assertEqual(
            search_status["constraints"]["banana_current"]["threshold"],
            1.6e4,
        )

    def test_single_stage_hardware_constraints_enforce_artifact_length_floor(self):
        module = load_single_stage_example_module()

        status = module.evaluate_single_stage_hardware_constraints(
            curve_curve_min_dist=0.05,
            cc_dist=0.05,
            curve_surface_min_dist=0.02,
            cs_dist=0.02,
            surface_vessel_min_dist=0.05,
            ss_dist=0.04,
            max_curvature=40.0,
            curvature_threshold=40.0,
            coil_length=0.8,
            length_target=1.9,
            poloidal_extent_rad=0.7,
            poloidal_extent_threshold_rad=0.8,
            tf_current_A=-8.0e4,
            tf_current_limit_A=8.0e4,
            banana_current_A=-1.6e4,
            banana_current_max_A=1.6e4,
            coil_width=0.12,
            width_min_threshold=0.05,
            width_max_threshold=0.17,
            self_intersect_penalty=0.0,
            self_intersect_threshold=0.0,
            lcfs_major_radius_m=in_bounds_lcfs_major_radius_m(),
            lcfs_minor_radius_m=in_bounds_lcfs_minor_radius_m(),
        )
        artifact_status = status["artifact_hardware_status"]

        self.assertFalse(artifact_status["success"])
        self.assertEqual(
            artifact_status["violations"],
            ["coil_length_min 0.800000 below threshold 0.950000"],
        )

    def test_surface_vessel_min_dist_uses_single_source_for_results(self):
        module = load_single_stage_example_module()

        class _Surface:
            def __init__(self, points):
                self._points = np.asarray(points, dtype=float).reshape((-1, 1, 3))

            def gamma(self):
                return self._points

        self.assertEqual(
            module.compute_single_stage_surface_vessel_min_dist(
                {"outer_vessel_gap": 0.456},
            ),
            0.456,
        )
        self.assertEqual(
            module.compute_single_stage_surface_vessel_min_dist(
                {"outer_vessel_gap": None},
                _Surface([[0.0, 0.0, 0.0]]),
                _Surface([[0.0, 0.3, 0.4]]),
            ),
            0.5,
        )

    def test_refinement_eligible_incumbent_requires_accepted_hardware_pass(self):
        module = load_single_stage_example_module()
        run_dict = {
            "accepted_x": np.array([1.0, 2.0]),
            "surface_state": {"sdofs": [np.array([1.0])], "iota": [0.15], "G": [1.0]},
            "J": 4.0,
            "dJ": np.array([1.0, -1.0]),
            "search_eval": {"total": 4.0},
            "surface_status": {"success": True},
            "search_surface_status": {"success": True},
            "accepted_hardware_status": {
                "success": False,
                "violations": ["coil_coil_min_dist"],
            },
            "topology_gate_status": {"enabled": False, "success": True},
            "intersecting": False,
        }

        self.assertFalse(module.refinement_eligible_incumbent(run_dict))

        run_dict["accepted_hardware_status"] = {"success": True, "violations": []}
        self.assertTrue(module.refinement_eligible_incumbent(run_dict))

    def test_refinement_eligible_incumbent_returns_python_bool(self):
        module = load_single_stage_example_module()
        run_dict = {
            "search_eval": {"total": 4.0},
            "surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "intersecting": False,
        }

        result = module.refinement_eligible_incumbent(run_dict)

        self.assertIs(type(result), bool)
        self.assertTrue(result)

    def test_write_json_artifact_normalizes_numpy_scalars(self):
        module = load_single_stage_example_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "results.json"
            module.write_json_artifact(
                str(artifact_path),
                {
                    "FINAL_FEASIBILITY_OK": np.bool_(True),
                    "SEARCH_OBJECTIVE_J": np.float64(1.25),
                    "INVALID_STATE_REJECTS_TOTAL": np.int64(3),
                    "FINAL_SEARCH_SURFACE_WEIGHTS": np.array([1.0, 0.5]),
                    "nested": {
                        "success": np.bool_(False),
                        "violations": [np.str_("cc")],
                    },
                },
            )

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))

        self.assertIs(payload["FINAL_FEASIBILITY_OK"], True)
        self.assertEqual(payload["SEARCH_OBJECTIVE_J"], 1.25)
        self.assertEqual(payload["INVALID_STATE_REJECTS_TOTAL"], 3)
        self.assertEqual(payload["FINAL_SEARCH_SURFACE_WEIGHTS"], [1.0, 0.5])
        self.assertIs(payload["nested"]["success"], False)
        self.assertEqual(payload["nested"]["violations"], ["cc"])

    def test_search_step_metrics_payload_defaults_to_zero(self):
        module = load_single_stage_example_module()

        payload = module.search_step_metrics_payload({})

        self.assertEqual(payload["SEARCH_STEP_EVALS"], 0)
        self.assertEqual(payload["SEARCH_STEP_REJECTED_AFTER_SURFACE_SOLVE"], 0)
        self.assertEqual(payload["SEARCH_STEP_FAST_OBJECTIVE_EVALS"], 0)
        self.assertEqual(payload["SEARCH_STEP_CURVATURE_REJECTS"], 0)
        self.assertEqual(payload["SEARCH_STEP_CURVATURE_PRECHECK_REJECTS"], 0)
        self.assertEqual(payload["SEARCH_STEP_CURVATURE_OVERCAP_BOOZER_EVALS"], 0)
        self.assertIsNone(payload["SEARCH_STEP_LAST_REJECTION_REASON"])

    def test_search_step_metrics_records_fast_curvature_reject(self):
        module = load_single_stage_example_module()
        run_dict = {}
        metrics = module.search_step_metrics_for_run(run_dict)
        metrics["evaluations"] += 1
        metrics["last_step_norm"] = 0.125

        module.record_search_step_objective_eval(
            metrics,
            {"total": 1.0, "grad": np.array([0.0]), "diagnostics_included": False},
        )
        module.record_search_step_rejection(
            metrics,
            rejection_reason="hardware",
            stack_status={"success": True},
            hardware_status={
                "success": False,
                "violations": [
                    "max_curvature 101.0 exceeds threshold 100.0",
                ],
                "constraints": {
                    "max_curvature": {"success": False},
                },
            },
            rejection_increment=2.5,
        )

        payload = module.search_step_metrics_payload(run_dict)

        self.assertEqual(payload["SEARCH_STEP_EVALS"], 1)
        self.assertEqual(payload["SEARCH_STEP_REJECTED_EVALS"], 1)
        self.assertEqual(payload["SEARCH_STEP_REJECTED_AFTER_SURFACE_SOLVE"], 1)
        self.assertEqual(payload["SEARCH_STEP_HARDWARE_REJECTS"], 1)
        self.assertEqual(payload["SEARCH_STEP_CURVATURE_REJECTS"], 1)
        self.assertEqual(payload["SEARCH_STEP_CURVATURE_PRECHECK_REJECTS"], 0)
        self.assertEqual(payload["SEARCH_STEP_CURVATURE_OVERCAP_BOOZER_EVALS"], 0)
        self.assertEqual(payload["SEARCH_STEP_FAST_OBJECTIVE_EVALS"], 1)
        self.assertEqual(payload["SEARCH_STEP_DIAGNOSTIC_OBJECTIVE_EVALS"], 0)
        self.assertEqual(payload["SEARCH_STEP_LAST_REJECTION_REASON"], "hardware")
        self.assertEqual(payload["SEARCH_STEP_LAST_STEP_NORM"], 0.125)
        self.assertEqual(payload["SEARCH_STEP_LAST_REJECTION_INCREMENT"], 2.5)

    def test_append_jsonl_artifact_rewrites_archive_without_partial_lines(self):
        module = load_single_stage_example_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "topology_archive.jsonl"
            archive_path.write_text('{"accepted_iteration": 0}\n', encoding="utf-8")

            module.append_jsonl_artifact(
                str(archive_path),
                {"accepted_iteration": np.int64(1), "weights": np.array([1.0, 0.5])},
            )

            payloads = [
                json.loads(line)
                for line in archive_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            payloads,
            [
                {"accepted_iteration": 0},
                {"accepted_iteration": 1, "weights": [1.0, 0.5]},
            ],
        )

    def test_record_residue_objective_diagnostics_writes_accepted_archive_entry(self):
        module = load_single_stage_example_module()
        residue_payload = {
            "schema_version": "greene_residue_objective_v1",
            "enabled": True,
            "target_manifest_id": "sha256:test-targets",
            "validation_id": "validation-artifact",
            "objective_weight": 0.25,
            "residue_scale": 0.5,
            "value": 0.125,
            "gradient_norm": 0.75,
            "branches": [
                {
                    "target_id": "p=0|q=1",
                    "branch": "O",
                    "residue": np.float64(0.25),
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            module.record_residue_objective_diagnostics(
                tmpdir,
                7,
                {
                    "residue_objective_payload": residue_payload,
                    "dJ_residue_objective": np.array([0.75]),
                },
            )
            archive_path = (
                Path(tmpdir) / module.GREENE_RESIDUE_OBJECTIVE_ARCHIVE_FILENAME
            )
            archive_entries = [
                json.loads(line)
                for line in archive_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(archive_entries), 1)
        self.assertEqual(
            archive_entries[0]["schema_version"],
            module.GREENE_RESIDUE_OBJECTIVE_ARCHIVE_SCHEMA_VERSION,
        )
        self.assertEqual(archive_entries[0]["accepted_iteration"], 7)
        self.assertEqual(
            archive_entries[0]["residue_objective"]["target_manifest_id"],
            "sha256:test-targets",
        )
        self.assertEqual(
            archive_entries[0]["residue_objective"]["branches"][0]["residue"],
            0.25,
        )

    def test_accepted_callback_writes_residue_objective_archive_entry(self):
        module = load_single_stage_example_module()

        class _Surface:
            nfp = 5

            def volume(self):
                return 1.0

            def gamma(self):
                return np.array([[[0.0, 0.0, 0.0]]])

            def unitnormal(self):
                return np.array([[[1.0, 0.0, 0.0]]])

            def save(self, path):
                self._saved_path = path

        class _ScalarObjective:
            def __init__(self, value):
                self._value = value

            def J(self):
                return self._value

            def dJ(self):
                return np.array([self._value, -self._value])

        class _DistanceObjective(_ScalarObjective):
            def __init__(self, value, min_distance):
                super().__init__(value)
                self._min_distance = min_distance

            def shortest_distance(self):
                return self._min_distance

        class _Curve:
            def gamma(self):
                return np.array([[1.0, 0.0, 0.0]])

        class _CurveLength:
            def J(self):
                return 1.7

        class _BiotSavart:
            def set_points(self, points):
                self._points = points

            def B(self):
                return np.array([[1.0, 0.0, 0.0]])

            def save(self, path):
                self._saved_path = path

        residue_payload = {
            "schema_version": "greene_residue_objective_v1",
            "enabled": True,
            "target_manifest_id": "sha256:test-targets",
            "validation_id": "validation-artifact",
            "objective_weight": 0.25,
            "residue_scale": 0.5,
            "value": 0.125,
            "branches": [{"target_id": "p=0|q=1", "branch": "O", "residue": 0.25}],
        }
        objective_eval = diagnostic_search_eval_payload(
            {
                "total": 1.25,
                "grad": np.array([0.1, -0.1]),
                "surface_weights": np.array([1.0]),
                "residue_objective_payload": residue_payload,
                "dJ_residue_objective": np.array([0.75]),
            }
        )
        surface = _Surface()
        surface_entry = {
            "name": "outer",
            "boozer_surface": SimpleNamespace(
                surface=surface,
                res={"success": True, "iota": TEST_IOTA, "G": TEST_G0},
                save=lambda path: None,
            ),
        }
        stack_status = {
            "success": True,
            "solve_success": [True],
            "self_intersections": [False],
            "volumes_ordered": True,
            "gap_ok": True,
            "nesting_ok": True,
            "adjacent_gaps": [],
            "outer_vessel_gap": None,
            "bad_nesting_phis": [],
        }
        hardware_snapshot = topology_hardware_snapshot()
        hardware_snapshot.update(
            {
                "curve_curve_min_dist": 0.06,
                "curve_surface_min_dist": 0.07,
                "surface_vessel_min_dist": 0.08,
                "max_curvature": 39.0,
                "length_target": 1.7,
                "tf_current_A": -8.0e4,
                "tf_current_limit_A": 8.0e4,
                "banana_current_A": 1.4e4,
                "banana_current_max_A": 1.6e4,
            }
        )
        module.surface_data = [surface_entry]
        module.outer_surface_data = surface_entry
        module.surface_iota_terms = [_ScalarObjective(TEST_IOTA)]
        module.JF = _ScalarObjective(0.0)
        module.JCurveLength = _ScalarObjective(0.44)
        module.JCurveCurve = _DistanceObjective(0.55, 0.06)
        module.JCurveSurface = _DistanceObjective(0.77, 0.07)
        module.JCurvature = _ScalarObjective(0.99)
        module.banana_curve = _Curve()
        module.banana_curves = [module.banana_curve]
        module.curvelength = _CurveLength()
        module.CurveLength = lambda curve: _CurveLength()
        module.bs = _BiotSavart()
        module.VV = object()
        module.CHECKPOINT_EVERY = 0
        module.TOPOLOGY_SCORER_EVERY = 0
        module.CONSTRAINT_METHOD = "penalty"
        module.CC_DIST = 0.05
        module.CS_DIST = 0.02
        module.PLASMA_VESSEL_MIN_DIST_M = 0.01
        module.CURVATURE_THRESHOLD = 40.0
        module.PRESERVED_TIMEOUT_REPLAY_CONFIG = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_10x10.nc",
            plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
            stage2_bs_path="",
            stage2_results_path="",
            mpol=0,
            ntor=0,
            nphi=0,
            ntheta=0,
            constraint_weight=None,
            constraint_method=None,
            alm_formulation=None,
            max_iterations=None,
            target_volume=None,
            target_iota=None,
        )
        module.stage = "initial"
        module.run_dict = {
            "surface_state": {"sdofs": [], "iota": [], "G": []},
            "accepted_x": np.zeros(2),
            "J": 0.0,
            "dJ": np.zeros(2),
            "search_eval": objective_eval,
            "surface_status": stack_status,
            "search_surface_status": stack_status,
            "accepted_hardware_status": hardware_snapshot["search_hardware_status"],
            "topology_gate_status": {"enabled": False, "success": True},
            "intersecting": False,
            "accepted_iterations": 3,
            "it": 4,
            "lscount": 0,
            "last_successful_eval": objective_eval,
            "last_successful_eval_weights": np.array([1.0]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            module.OUT_DIR_ITER = tmpdir
            with (
                patch.object(
                    module,
                    "snapshot_surface_states",
                    return_value={"sdofs": [], "iota": [], "G": []},
                ),
                patch.object(
                    module, "evaluate_surface_stack", return_value=stack_status
                ),
                patch.object(
                    module,
                    "evaluate_single_stage_hardware_snapshot",
                    return_value=hardware_snapshot,
                ),
                patch.object(module, "maybe_record_topology_score", return_value=None),
                patch.object(
                    module,
                    "compute_surface_field_metrics",
                    return_value=(0.0, 0.0),
                ),
            ):
                module.callback(np.array([0.25, -0.25]))
            archive_path = (
                Path(tmpdir) / module.GREENE_RESIDUE_OBJECTIVE_ARCHIVE_FILENAME
            )
            archive_entries = [
                json.loads(line)
                for line in archive_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(archive_entries[0]["accepted_iteration"], 4)
        self.assertEqual(
            archive_entries[0]["residue_objective"]["target_manifest_id"],
            "sha256:test-targets",
        )

    def test_topology_archive_entry_preserves_kam_metrics(self):
        module = load_single_stage_example_module()
        topology_result = {
            "evaluation_state": "evaluated",
            "broken": False,
            "evaluation_error": None,
            "evaluation_error_type": None,
            "survival_fraction": 1.0,
            "survived_lines": 12,
            "nfieldlines": 12,
            "tmax": 80.0,
            "mean_exit_time": None,
            "confinement_score": 1.0,
            "mean_line_loss": 0.0,
            "worst_k_line_loss": 0.0,
            "early_exit_fraction": 0.0,
            "confinement_loss": 0.0,
            "confinement_surrogate_k": 3,
            "confinement_early_exit_threshold": 0.2,
            "invariant_torus_fraction": 1.0 / 12.0,
            "kam_fraction": 1.0 / 12.0,
            "wba_fraction_denominator_policy": "survived_non_lost_seeds",
            "wba_fraction_denominator_seed_count": 12,
            "kam_median_width": 0.08275987797445103,
            "cross_section_span": 0.19712474791042184,
            "stop_reason_counts": {"surface_exit": 0},
            "first_exit": None,
            "per_phi_hit_counts": [71, 60, 60, 60],
            "line_metrics": [],
            "line_lifetimes": [],
            "line_losses": [],
            "seed_contract": {"nfieldlines": 12},
            "field_model": "biotsavart",
            "transport_diagnostics": {"status": "partial"},
        }

        entry = module.topology_archive_entry(
            9,
            -0.701188557808912,
            -0.701188557808912,
            topology_result,
            topology_hardware_snapshot(),
        )

        self.assertEqual(entry["accepted_iteration"], 9)
        self.assertEqual(entry["topology_archive_schema_version"], 2)
        self.assertEqual(
            entry["invariant_torus_fraction"],
            topology_result["invariant_torus_fraction"],
        )
        self.assertEqual(entry["kam_fraction"], topology_result["kam_fraction"])
        self.assertEqual(
            entry["wba_fraction_denominator_policy"],
            topology_result["wba_fraction_denominator_policy"],
        )
        self.assertEqual(
            entry["wba_fraction_denominator_seed_count"],
            topology_result["wba_fraction_denominator_seed_count"],
        )
        self.assertEqual(entry["kam_median_width"], topology_result["kam_median_width"])
        self.assertEqual(
            entry["cross_section_span"], topology_result["cross_section_span"]
        )
        self.assertTrue(entry["artifact_hardware_ok"])
        self.assertTrue(entry["search_hardware_ok"])

    def test_initial_topology_score_writes_archive_and_confinement_checkpoint(self):
        module = load_single_stage_example_module()
        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_INVARIANT_TORUS_MIN = 0.30
        module.FRONTIER_KAM_MIN = 0.30
        module.TOPOLOGY_SCORER_EVERY = 5
        module.TOPOLOGY_SCORER_NFIELDLINES = 12
        module.TOPOLOGY_SCORER_TMAX = 80.0
        module.CONFINEMENT_OBJECTIVE_WEIGHT = 1.0
        module.CONFINEMENT_SURROGATE_WORST_K = 3
        module.CONFINEMENT_SURROGATE_EARLY_THRESHOLD = 0.2
        module.CONFINEMENT_SURROGATE_MEAN_WEIGHT = 1.0
        module.CONFINEMENT_SURROGATE_WORST_WEIGHT = 0.0
        module.CONFINEMENT_SURROGATE_EARLY_WEIGHT = 0.0
        topology_result = {
            "evaluation_state": "evaluated",
            "broken": False,
            "evaluation_error": None,
            "evaluation_error_type": None,
            "survival_fraction": 1.0,
            "survived_lines": 12,
            "nfieldlines": 12,
            "tmax": 80.0,
            "mean_exit_time": None,
            "confinement_score": 4.0,
            "mean_line_loss": 0.0,
            "worst_k_line_loss": 0.0,
            "early_exit_fraction": 0.0,
            "confinement_loss": 0.25,
            "confinement_surrogate_k": 3,
            "confinement_early_exit_threshold": 0.2,
            "invariant_torus_fraction": 0.5,
            "kam_fraction": 0.5,
            "kam_median_width": 0.08,
            "cross_section_span": 0.2,
            "stop_reason_counts": {"surface_exit": 0},
            "first_exit": None,
            "per_phi_hit_counts": [60, 60, 60, 60],
            "line_metrics": [],
            "line_lifetimes": [],
            "line_losses": [],
            "seed_contract": {"nfieldlines": 12},
            "field_model": "biotsavart",
            "transport_diagnostics": {"status": "partial"},
        }
        run_dict = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            module.OUT_DIR_ITER = tmpdir
            with (
                patch.object(
                    module,
                    "safe_score_topology",
                    return_value=topology_result,
                ),
                patch.object(
                    module, "write_topology_checkpoint_artifacts"
                ) as write_checkpoint,
            ):
                entry = module.maybe_record_topology_score(
                    run_dict,
                    accepted_iteration=0,
                    proxy_objective=2.0,
                    outer_surf=object(),
                    biotsavart=object(),
                    surface_data=[],
                    hardware_snapshot=topology_hardware_snapshot(),
                )

            archive_entries = [
                json.loads(line)
                for line in (Path(tmpdir) / "topology_archive.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(entry["accepted_iteration"], 0)
        self.assertEqual(archive_entries[0]["accepted_iteration"], 0)
        self.assertEqual(archive_entries[0]["topology_archive_schema_version"], 2)
        self.assertTrue(archive_entries[0]["artifact_hardware_ok"])
        self.assertEqual(run_dict["best_topology"]["accepted_iteration"], 0)
        self.assertEqual(run_dict["best_hw_clean_topology"]["accepted_iteration"], 0)
        self.assertEqual(
            run_dict["best_confinement_objective"]["accepted_iteration"], 0
        )
        checkpoint_dir_names = [
            Path(call.args[0]).name for call in write_checkpoint.call_args_list
        ]
        self.assertIn("best_topology", checkpoint_dir_names)
        self.assertIn("best_hw_clean_topology", checkpoint_dir_names)
        self.assertIn("best_confinement_objective", checkpoint_dir_names)

    def test_best_topology_ranks_survival_before_kam(self):
        module = load_single_stage_example_module()
        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_INVARIANT_TORUS_MIN = 0.30
        module.FRONTIER_KAM_MIN = 0.30
        module.TOPOLOGY_SCORER_EVERY = 1
        module.TOPOLOGY_SCORER_NFIELDLINES = 12
        module.TOPOLOGY_SCORER_TMAX = 80.0
        module.CONFINEMENT_OBJECTIVE_WEIGHT = 1.0
        module.CONFINEMENT_SURROGATE_WORST_K = 3
        module.CONFINEMENT_SURROGATE_EARLY_THRESHOLD = 0.2
        module.CONFINEMENT_SURROGATE_MEAN_WEIGHT = 1.0
        module.CONFINEMENT_SURROGATE_WORST_WEIGHT = 0.0
        module.CONFINEMENT_SURROGATE_EARLY_WEIGHT = 0.0

        def topology_result(kam_fraction, survival_fraction, confinement_score):
            return {
                "evaluation_state": "evaluated",
                "broken": False,
                "evaluation_error": None,
                "evaluation_error_type": None,
                "survival_fraction": survival_fraction,
                "survived_lines": int(round(12 * survival_fraction)),
                "nfieldlines": 12,
                "tmax": 80.0,
                "mean_exit_time": None,
                "confinement_score": confinement_score,
                "mean_line_loss": 0.0,
                "worst_k_line_loss": 0.0,
                "early_exit_fraction": 0.0,
                "confinement_loss": 1.0 / confinement_score,
                "confinement_surrogate_k": 3,
                "confinement_early_exit_threshold": 0.2,
                "invariant_torus_fraction": kam_fraction,
                "kam_fraction": kam_fraction,
                "kam_median_width": 0.08,
                "cross_section_span": 0.2,
                "stop_reason_counts": {"surface_exit": 0},
                "first_exit": None,
                "per_phi_hit_counts": [60, 60, 60, 60],
                "line_metrics": [],
                "line_lifetimes": [],
                "line_losses": [],
                "seed_contract": {"nfieldlines": 12},
                "field_model": "biotsavart",
                "transport_diagnostics": {"status": "partial"},
            }

        run_dict = {}
        low_kam_high_confinement = topology_result(
            kam_fraction=1.0 / 12.0,
            survival_fraction=1.0,
            confinement_score=100.0,
        )
        high_kam_lower_confinement = topology_result(
            kam_fraction=0.5,
            survival_fraction=0.75,
            confinement_score=2.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            module.OUT_DIR_ITER = tmpdir
            with (
                patch.object(
                    module,
                    "safe_score_topology",
                    side_effect=[low_kam_high_confinement, high_kam_lower_confinement],
                ),
                patch.object(module, "write_topology_checkpoint_artifacts"),
            ):
                first_entry = module.maybe_record_topology_score(
                    run_dict,
                    accepted_iteration=1,
                    proxy_objective=5.0,
                    outer_surf=object(),
                    biotsavart=object(),
                    surface_data=[],
                    hardware_snapshot=topology_hardware_snapshot(),
                )
                second_entry = module.maybe_record_topology_score(
                    run_dict,
                    accepted_iteration=2,
                    proxy_objective=5.0,
                    outer_surf=object(),
                    biotsavart=object(),
                    surface_data=[],
                    hardware_snapshot=topology_hardware_snapshot(),
                )

            archive_entries = [
                json.loads(line)
                for line in (Path(tmpdir) / "topology_archive.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertFalse(first_entry["frontier_certification_ok"])
        self.assertTrue(second_entry["frontier_certification_ok"])
        self.assertEqual(run_dict["best_topology"]["accepted_iteration"], 1)
        self.assertEqual(archive_entries[0]["accepted_iteration"], 1)
        self.assertEqual(archive_entries[1]["accepted_iteration"], 2)
        payload = module.best_topology_results_payload(run_dict)
        self.assertAlmostEqual(
            payload["BEST_TOPOLOGY_INVARIANT_TORUS_FRACTION"],
            1.0 / 12.0,
        )
        self.assertAlmostEqual(payload["BEST_TOPOLOGY_KAM_FRACTION"], 1.0 / 12.0)
        self.assertEqual(
            payload["BEST_TOPOLOGY_CERTIFICATION_REASON"],
            "invariant_torus_fraction_below_min",
        )

    def test_best_hw_clean_topology_filters_strict_artifact_hardware(self):
        module = load_single_stage_example_module()
        module.SINGLE_STAGE_GOAL_MODE = "target"
        module.FRONTIER_INVARIANT_TORUS_MIN = 0.30
        module.FRONTIER_KAM_MIN = 0.30
        module.TOPOLOGY_SCORER_EVERY = 1
        module.TOPOLOGY_SCORER_NFIELDLINES = 8
        module.TOPOLOGY_SCORER_TMAX = 25.0
        module.CONFINEMENT_OBJECTIVE_WEIGHT = 1.0
        module.CONFINEMENT_SURROGATE_WORST_K = 3
        module.CONFINEMENT_SURROGATE_EARLY_THRESHOLD = 0.2
        module.CONFINEMENT_SURROGATE_MEAN_WEIGHT = 1.0
        module.CONFINEMENT_SURROGATE_WORST_WEIGHT = 0.0
        module.CONFINEMENT_SURROGATE_EARLY_WEIGHT = 0.0

        def topology_result(kam_fraction, survival_fraction, confinement_score):
            return {
                "evaluation_state": "evaluated",
                "broken": False,
                "evaluation_error": None,
                "evaluation_error_type": None,
                "survival_fraction": survival_fraction,
                "survived_lines": int(round(8 * survival_fraction)),
                "nfieldlines": 8,
                "tmax": 25.0,
                "mean_exit_time": None,
                "confinement_score": confinement_score,
                "mean_line_loss": 0.0,
                "worst_k_line_loss": 0.0,
                "early_exit_fraction": 0.0,
                "confinement_loss": 1.0 / confinement_score,
                "confinement_surrogate_k": 3,
                "confinement_early_exit_threshold": 0.2,
                "invariant_torus_fraction": kam_fraction,
                "kam_fraction": kam_fraction,
                "kam_median_width": 0.08,
                "cross_section_span": 0.2,
                "stop_reason_counts": {"surface_exit": 0},
                "first_exit": None,
                "per_phi_hit_counts": [40, 40, 40, 40],
                "line_metrics": [],
                "line_lifetimes": [],
                "line_losses": [],
                "seed_contract": {"nfieldlines": 8},
                "field_model": "biotsavart",
                "transport_diagnostics": {"status": "partial"},
            }

        run_dict = {}
        hw_clean_lower_kam = topology_result(
            kam_fraction=0.375,
            survival_fraction=0.875,
            confinement_score=0.899,
        )
        hw_failed_higher_kam = topology_result(
            kam_fraction=0.625,
            survival_fraction=1.0,
            confinement_score=1.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            module.OUT_DIR_ITER = tmpdir
            with (
                patch.object(
                    module,
                    "safe_score_topology",
                    side_effect=[hw_clean_lower_kam, hw_failed_higher_kam],
                ),
                patch.object(module, "write_topology_checkpoint_artifacts"),
            ):
                first_entry = module.maybe_record_topology_score(
                    run_dict,
                    accepted_iteration=0,
                    proxy_objective=3.0,
                    outer_surf=object(),
                    biotsavart=object(),
                    surface_data=[],
                    hardware_snapshot=topology_hardware_snapshot(),
                )
                second_entry = module.maybe_record_topology_score(
                    run_dict,
                    accepted_iteration=3,
                    proxy_objective=2.5,
                    outer_surf=object(),
                    biotsavart=object(),
                    surface_data=[],
                    hardware_snapshot=topology_hardware_snapshot(
                        artifact_success=False,
                        artifact_violations=[
                            "max_curvature 102.000000 above threshold 100.000000"
                        ],
                    ),
                )

            archive_entries = [
                json.loads(line)
                for line in (Path(tmpdir) / "topology_archive.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertTrue(first_entry["artifact_hardware_ok"])
        self.assertFalse(second_entry["artifact_hardware_ok"])
        self.assertEqual(run_dict["best_topology"]["accepted_iteration"], 3)
        self.assertEqual(run_dict["best_hw_clean_topology"]["accepted_iteration"], 0)
        self.assertFalse(archive_entries[1]["artifact_hardware_ok"])
        payload = module.best_topology_results_payload(run_dict)
        self.assertAlmostEqual(payload["BEST_TOPOLOGY_INVARIANT_TORUS_FRACTION"], 0.625)
        self.assertAlmostEqual(payload["BEST_TOPOLOGY_KAM_FRACTION"], 0.625)
        self.assertFalse(payload["BEST_TOPOLOGY_ARTIFACT_HARDWARE_OK"])
        self.assertAlmostEqual(
            payload["BEST_HW_CLEAN_TOPOLOGY_INVARIANT_TORUS_FRACTION"],
            0.375,
        )
        self.assertAlmostEqual(payload["BEST_HW_CLEAN_TOPOLOGY_KAM_FRACTION"], 0.375)
        self.assertTrue(payload["BEST_HW_CLEAN_TOPOLOGY_ARTIFACT_HARDWARE_OK"])

    def test_best_topology_payload_keeps_missing_records_unknown(self):
        module = load_single_stage_example_module()

        empty_payload = module.best_topology_results_payload({})

        self.assertIsNone(empty_payload["BEST_TOPOLOGY_KAM_FRACTION"])
        self.assertIsNone(empty_payload["BEST_TOPOLOGY_ARTIFACT_HARDWARE_OK"])
        self.assertIsNone(empty_payload["BEST_TOPOLOGY_DIAGNOSTICS"])
        self.assertIsNone(empty_payload["BEST_HW_CLEAN_TOPOLOGY_KAM_FRACTION"])
        self.assertIsNone(empty_payload["BEST_HW_CLEAN_TOPOLOGY_ARTIFACT_HARDWARE_OK"])
        self.assertIsNone(empty_payload["BEST_HW_CLEAN_TOPOLOGY_DIAGNOSTICS"])

        raw_only_payload = module.best_topology_results_payload(
            {
                "best_topology": {
                    "accepted_iteration": 3,
                    "topology_broken": False,
                    "kam_fraction": 0.625,
                    "artifact_hardware_ok": False,
                }
            }
        )

        self.assertAlmostEqual(raw_only_payload["BEST_TOPOLOGY_KAM_FRACTION"], 0.625)
        self.assertFalse(raw_only_payload["BEST_TOPOLOGY_ARTIFACT_HARDWARE_OK"])
        self.assertEqual(
            raw_only_payload["BEST_TOPOLOGY_DIAGNOSTICS"]["reason"],
            "score_recorded",
        )
        self.assertIsNone(raw_only_payload["BEST_HW_CLEAN_TOPOLOGY_KAM_FRACTION"])
        self.assertIsNone(raw_only_payload["BEST_HW_CLEAN_TOPOLOGY_DIAGNOSTICS"])

        unproven_hw_clean_payload = module.best_topology_results_payload(
            {
                "best_hw_clean_topology": {
                    "accepted_iteration": 4,
                    "topology_broken": False,
                    "kam_fraction": 0.75,
                }
            }
        )

        self.assertIsNone(
            unproven_hw_clean_payload["BEST_HW_CLEAN_TOPOLOGY_KAM_FRACTION"]
        )
        self.assertIsNone(
            unproven_hw_clean_payload["BEST_HW_CLEAN_TOPOLOGY_ARTIFACT_HARDWARE_OK"]
        )

    def test_legacy_resume_clears_pre_resume_topology_entries(self):
        module = load_single_stage_example_module()
        run_dict = {
            "latest_topology_entry": {"accepted_iteration": 0},
            "best_topology": {"accepted_iteration": 0},
            "best_hw_clean_topology": {"accepted_iteration": 0},
            "best_confinement_objective": {"accepted_iteration": 0},
        }

        module.restore_topology_checkpoint_entries(
            run_dict,
            {
                "best_topology": {
                    "accepted_iteration": 7,
                    "kam_fraction": 0.5,
                },
                "best_hw_clean_topology": {
                    "accepted_iteration": 8,
                    "kam_fraction": 0.75,
                },
            },
        )

        self.assertNotIn("latest_topology_entry", run_dict)
        self.assertNotIn("best_hw_clean_topology", run_dict)
        self.assertNotIn("best_confinement_objective", run_dict)
        self.assertEqual(run_dict["best_topology"]["accepted_iteration"], 7)

    def test_build_banana_current_coordinate_report_tracks_indices_bounds_and_gradients(
        self,
    ):
        module = load_single_stage_example_module()
        current_a, current_b, banana_current_state, objective = (
            build_banana_current_report_fixture(module)
        )
        x = np.array([0.25, 1.6e4, -0.5, -1.6e4], dtype=float)
        grad = np.array([2.0, 3.0e-4, -4.0, -4.0e-4], dtype=float)

        report = module.build_banana_current_coordinate_report(
            objective,
            banana_current_state,
            x,
            grad,
            phase="accepted",
            accepted_iteration=7,
        )

        self.assertEqual(report["phase"], "accepted")
        self.assertEqual(report["accepted_iteration"], 7)
        self.assertEqual(report["coordinate_indices"], [1, 3])
        self.assertEqual(
            report["coordinate_dof_names"],
            [*current_a.dof_names, *current_b.dof_names],
        )
        self.assertEqual(report["coordinate_values_A"], [1.6e4, -1.6e4])
        self.assertEqual(report["coordinate_gradients"], [3.0e-4, -4.0e-4])
        self.assertTrue(report["bound_activity"][0]["at_upper_bound"])
        self.assertTrue(report["bound_activity"][1]["at_lower_bound"])
        self.assertAlmostEqual(
            report["coordinate_gradient_l2"],
            float(np.linalg.norm([3.0e-4, -4.0e-4])),
        )
        self.assertAlmostEqual(
            report["noncurrent_gradient_l2"],
            float(np.linalg.norm([2.0, -4.0])),
        )

    def test_build_banana_current_coordinate_report_prefers_active_optimizer_bounds(
        self,
    ):
        module = load_single_stage_example_module()
        _, _, banana_current_state, objective = build_banana_current_report_fixture(
            module
        )
        x = np.array([0.25, 1.505e4, -0.5, -1.505e4], dtype=float)
        active_bounds = [
            (-1.0, 1.0),
            (1.495e4, 1.505e4),
            (-1.0, 1.0),
            (-1.505e4, -1.495e4),
        ]

        report = module.build_banana_current_coordinate_report(
            objective,
            banana_current_state,
            x,
            grad=None,
            active_optimizer_bounds=active_bounds,
            phase="accepted",
            accepted_iteration=3,
        )

        self.assertEqual(report["coordinate_lower_bounds_A"], [1.495e4, -1.505e4])
        self.assertEqual(report["coordinate_upper_bounds_A"], [1.505e4, -1.495e4])
        self.assertTrue(report["bound_activity"][0]["at_upper_bound"])
        self.assertTrue(report["bound_activity"][1]["at_lower_bound"])

    def test_build_banana_current_coordinate_report_tracks_projected_gradients(self):
        module = load_single_stage_example_module()
        _, _, banana_current_state, objective = build_banana_current_report_fixture(
            module
        )
        x = np.array([0.25, 1.6e4, -0.5, -1.6e4], dtype=float)
        grad = np.array([2.0, -3.0e-4, -4.0, 4.0e-4], dtype=float)

        report = module.build_banana_current_coordinate_report(
            objective,
            banana_current_state,
            x,
            grad,
            phase="accepted",
            accepted_iteration=7,
        )

        self.assertEqual(report["coordinate_gradients"], [-3.0e-4, 4.0e-4])
        self.assertEqual(report["projected_coordinate_gradients"], [0.0, 0.0])
        self.assertEqual(report["projected_coordinate_gradient_l2"], 0.0)
        self.assertAlmostEqual(
            report["projected_noncurrent_gradient_l2"],
            float(np.linalg.norm([2.0, -4.0])),
        )
        self.assertEqual(
            report["projected_coordinate_to_noncurrent_gradient_ratio"], 0.0
        )

    def test_build_banana_current_coordinate_report_separates_scaled_optimizer_units_from_amps(
        self,
    ):
        module = load_single_stage_example_module()
        current_a = ScaledCurrent(Current(1.5), 1.0e4)
        current_b = ScaledCurrent(Current(-2.0), 8.0e3)
        banana_current_state = module.SingleStageBananaCurrentState(
            mode="independent",
            currents=(current_a, current_b),
            seed_currents_A=(1.5e4, -1.6e4),
            coordinate_scaling="seed-relative",
            current_coordinate_scale_factors_A=(1.0e4, 8.0e3),
        )
        objective = SimpleNamespace(
            dof_names=("geom:0", *current_a.dof_names, "geom:1", *current_b.dof_names),
            lower_bounds=np.array([-1.0, -1.6, -1.0, -2.0], dtype=float),
            upper_bounds=np.array([1.0, 1.6, 1.0, 2.0], dtype=float),
        )
        x = np.array([0.25, 1.5, -0.5, -2.0], dtype=float)
        grad = np.array([2.0, 3.0, -4.0, 4.0], dtype=float)

        report = module.build_banana_current_coordinate_report(
            objective,
            banana_current_state,
            x,
            grad,
            phase="accepted",
            accepted_iteration=7,
        )

        self.assertEqual(report["coordinate_values_A"], [1.5e4, -1.6e4])
        self.assertEqual(report["coordinate_lower_bounds_A"], [-1.6e4, -1.6e4])
        self.assertEqual(report["coordinate_upper_bounds_A"], [1.6e4, 1.6e4])
        self.assertEqual(report["optimizer_coordinate_values"], [1.5, -2.0])
        self.assertEqual(report["optimizer_coordinate_lower_bounds"], [-1.6, -2.0])
        self.assertEqual(report["optimizer_coordinate_upper_bounds"], [1.6, 2.0])
        self.assertEqual(report["current_coordinate_scale_factors_A"], [1.0e4, 8.0e3])
        self.assertEqual(report["coordinate_gradients"], [3.0e-4, 5.0e-4])
        self.assertAlmostEqual(
            report["full_gradient_l2"],
            float(np.linalg.norm([2.0, 3.0e-4, -4.0, 5.0e-4])),
        )
        self.assertEqual(report["optimizer_coordinate_gradients"], [3.0, 4.0])
        self.assertAlmostEqual(report["optimizer_coordinate_gradient_l2"], 5.0)
        self.assertEqual(report["projected_optimizer_coordinate_gradients"], [3.0, 0.0])
        self.assertEqual(report["projected_coordinate_gradients"], [3.0e-4, 0.0])
        self.assertFalse(report["bound_activity"][0]["at_upper_bound"])
        self.assertTrue(report["bound_activity"][1]["at_lower_bound"])

    def test_write_banana_current_diagnostics_artifact_serializes_seed_and_recent_rejects(
        self,
    ):
        module = load_single_stage_example_module()
        _, _, banana_current_state, objective = build_banana_current_report_fixture(
            module
        )
        seed_x = np.array([0.1, 1.5e4, -0.2, -1.5e4], dtype=float)
        seed_grad = np.array([1.0, 2.0e-4, -2.0, -3.0e-4], dtype=float)
        rejected_x = np.array([0.1, 1.6e4, -0.2, -1.4e4], dtype=float)
        rejected_grad = np.array([1.0, 4.0e-4, -2.0, -1.0e-4], dtype=float)

        def finite_difference_probe_fn(trial_x):
            return {
                "objective_total": float(np.sum(np.asarray(trial_x, dtype=float) ** 2)),
                "hardware_ok": True,
                "topology_state": "ok",
            }

        diagnostics_state = module.build_banana_current_diagnostics_state(
            objective,
            banana_current_state,
            seed_x,
            seed_grad,
            accepted_iteration=0,
            baseline_total=float(np.sum(seed_x**2)),
            finite_difference_probe_fn=finite_difference_probe_fn,
            finite_difference_relative_step_fraction=0.01,
        )
        rejected_report = module.build_banana_current_coordinate_report(
            objective,
            banana_current_state,
            rejected_x,
            rejected_grad,
            phase="rejected_trial",
            accepted_iteration=0,
            line_search_evaluations=3,
            step_norm=0.5,
            rejection_reason="hardware",
        )
        module.record_banana_current_diagnostics_report(
            diagnostics_state,
            rejected_report,
            rejected_trial=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            module.write_banana_current_diagnostics_artifact(
                tmpdir,
                diagnostics_state,
            )
            payload = json.loads(
                (Path(tmpdir) / module.BANANA_CURRENT_DIAGNOSTICS_FILENAME).read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(
            payload["schema_version"],
            module.SINGLE_STAGE_BANANA_CURRENT_DIAGNOSTICS_SCHEMA_VERSION,
        )
        self.assertEqual(payload["seed_report"]["delta_from_seed_A"], [0.0, 0.0])
        self.assertIsNotNone(payload["seed_finite_difference_probe"])
        self.assertEqual(
            payload["seed_finite_difference_probe"]["noncurrent_selection"],
            module.BANANA_CURRENT_FD_NONCURRENT_SELECTION,
        )
        self.assertEqual(payload["rejected_trial_reports_recorded"], 1)
        self.assertEqual(
            payload["latest_rejected_trial_report"]["rejection_reason"], "hardware"
        )
        self.assertEqual(
            payload["latest_rejected_trial_report"]["delta_from_seed_A"],
            [1.0e3, 1.0e3],
        )
        self.assertEqual(len(payload["recent_rejected_trial_reports"]), 1)

    def test_build_banana_current_diagnostics_state_anchors_seed_to_effective_start(
        self,
    ):
        module = load_single_stage_example_module()
        _, _, banana_current_state, objective = build_banana_current_report_fixture(
            module,
            seed_currents_A=(1.45e4, -1.45e4),
        )
        seed_x = np.array([0.1, 1.5e4, -0.2, -1.5e4], dtype=float)
        seed_grad = np.array([1.0, 2.0e-4, -2.0, -3.0e-4], dtype=float)

        diagnostics_state = module.build_banana_current_diagnostics_state(
            objective,
            banana_current_state,
            seed_x,
            seed_grad,
            accepted_iteration=4,
        )

        self.assertEqual(diagnostics_state["seed_currents_A"], [1.5e4, -1.5e4])
        self.assertEqual(
            diagnostics_state["configured_seed_currents_A"],
            [1.45e4, -1.45e4],
        )
        self.assertEqual(
            diagnostics_state["seed_report"]["coordinate_values_A"],
            diagnostics_state["seed_currents_A"],
        )
        self.assertEqual(
            diagnostics_state["seed_report"]["delta_from_seed_A"],
            [0.0, 0.0],
        )

    def test_build_banana_current_finite_difference_probe_matches_noncurrent_coordinates(
        self,
    ):
        module = load_single_stage_example_module()
        _, _, banana_current_state, objective = build_banana_current_report_fixture(
            module
        )
        baseline_x = np.array([0.5, 1.5e4, -0.25, -1.5e4], dtype=float)

        def objective_probe_fn(trial_x):
            trial_x = np.asarray(trial_x, dtype=float)
            return {
                "objective_total": float(
                    trial_x[0] ** 2
                    + 4.0 * trial_x[2] ** 2
                    + 1.0e-6 * trial_x[1] ** 2
                    + 2.0e-6 * trial_x[3] ** 2
                ),
                "hardware_ok": True,
                "topology_state": "ok",
            }

        probe = module.build_banana_current_finite_difference_probe(
            objective,
            banana_current_state,
            baseline_x,
            baseline_total=float(objective_probe_fn(baseline_x)["objective_total"]),
            objective_probe_fn=objective_probe_fn,
            relative_step_fraction=0.01,
        )

        self.assertEqual(probe["current_group"]["coordinate_indices"], [1, 3])
        self.assertEqual(
            probe["matched_noncurrent_group"]["coordinate_indices"],
            [0, 2],
        )
        self.assertEqual(
            probe["matched_noncurrent_group"]["selection"],
            module.BANANA_CURRENT_FD_NONCURRENT_SELECTION,
        )
        self.assertEqual(
            probe["current_group"]["summary"]["successful_coordinate_count"],
            2,
        )
        self.assertEqual(
            probe["matched_noncurrent_group"]["summary"]["successful_coordinate_count"],
            2,
        )
        self.assertIsNotNone(
            probe["comparison"][
                "current_to_noncurrent_mean_max_abs_objective_delta_ratio"
            ]
        )

    def test_build_banana_current_finite_difference_probe_clips_steps_to_bounds(self):
        module = load_single_stage_example_module()
        _, _, banana_current_state, objective = build_banana_current_report_fixture(
            module
        )
        baseline_x = np.array([0.25, 1.599e4, -0.5, -1.599e4], dtype=float)

        def objective_probe_fn(trial_x):
            trial_x = np.asarray(trial_x, dtype=float)
            return float(np.sum(trial_x**2))

        probe = module.build_banana_current_finite_difference_probe(
            objective,
            banana_current_state,
            baseline_x,
            baseline_total=float(objective_probe_fn(baseline_x)),
            objective_probe_fn=objective_probe_fn,
            relative_step_fraction=0.01,
        )

        first_current_sample = probe["current_group"]["samples"][0]
        self.assertTrue(first_current_sample["step_clipped_to_bounds"])
        self.assertAlmostEqual(first_current_sample["requested_step"], 159.9)
        self.assertAlmostEqual(first_current_sample["applied_step"], 10.0)

    def test_normalize_optimizer_termination_message_enriches_blank_abnormal(self):
        module = load_single_stage_example_module()

        message = module.normalize_optimizer_termination_message(
            "ABNORMAL: ",
            success=False,
            status=np.int64(8),
            invalid_state_rejects_total=np.int64(29),
            surface_solve_rejects=np.int64(29),
            hardware_rejects=np.int64(0),
            topology_gate_rejects=np.int64(0),
        )

        self.assertEqual(
            message,
            "ABNORMAL: empty SciPy L-BFGS-B task; status=8; "
            "invalid_state_rejects=29; surface_solve_rejects=29; "
            "hardware_rejects=0; topology_gate_rejects=0",
        )

    def test_normalize_optimizer_termination_message_decodes_bytes_abnormal(self):
        module = load_single_stage_example_module()

        message = module.normalize_optimizer_termination_message(
            b"ABNORMAL: ",
            success=False,
            status=8,
            invalid_state_rejects_total=11,
        )

        self.assertEqual(
            message,
            "ABNORMAL: empty SciPy L-BFGS-B task; status=8; invalid_state_rejects=11",
        )

    def test_normalize_optimizer_termination_message_preserves_non_abnormal_text(self):
        module = load_single_stage_example_module()

        message = module.normalize_optimizer_termination_message(
            "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
            success=False,
            status=5,
        )

        self.assertEqual(message, "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT")

    def test_maybe_update_best_feasible_incumbent_uses_search_total_metric(self):
        module = load_single_stage_example_module()
        run_dict = {
            "accepted_x": np.array([1.0, 2.0]),
            "surface_state": {"sdofs": [np.array([1.0])], "iota": [0.15], "G": [1.0]},
            "J": 4.0,
            "dJ": np.array([1.0, -1.0]),
            "search_eval": {"total": 4.0, "surface_weights": np.array([1.0])},
            "surface_status": {"success": True},
            "search_surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"enabled": False, "success": True},
            "intersecting": False,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
        }

        def rich_eval_for_current_state(surface_weights, *, include_diagnostics=None):
            self.assertTrue(include_diagnostics)
            np.testing.assert_allclose(
                surface_weights,
                run_dict["search_eval"]["surface_weights"],
            )
            return diagnostic_search_eval_payload(run_dict["search_eval"])

        with patch.object(
            module,
            "evaluate_search_objective",
            side_effect=rich_eval_for_current_state,
        ) as evaluate_mock:
            self.assertTrue(
                module.maybe_update_best_feasible_incumbent(run_dict, "initial")
            )
            self.assertEqual(evaluate_mock.call_count, 1)

            self.assertEqual(run_dict["best_feasible_metric"], 4.0)
            self.assertEqual(run_dict["best_feasible_stage"], "initial")
            np.testing.assert_allclose(
                run_dict["best_feasible_incumbent"].x,
                [1.0, 2.0],
            )
            self.assertNotIn("J_QS", run_dict["search_eval"])
            self.assertAlmostEqual(
                run_dict["best_feasible_incumbent"].search_eval["J_QS"],
                2.5e-4,
            )

            run_dict["search_eval"] = {
                "total": 5.0,
                "surface_weights": np.array([1.0]),
            }
            run_dict["J"] = 5.0
            self.assertFalse(
                module.maybe_update_best_feasible_incumbent(run_dict, "final")
            )
            self.assertEqual(evaluate_mock.call_count, 1)
            self.assertEqual(run_dict["best_feasible_metric"], 4.0)
            self.assertEqual(run_dict["best_feasible_stage"], "initial")

            run_dict["search_eval"] = {
                "total": 3.0,
                "surface_weights": np.array([1.0]),
            }
            run_dict["J"] = 3.0
            self.assertTrue(
                module.maybe_update_best_feasible_incumbent(run_dict, "final")
            )
            self.assertEqual(evaluate_mock.call_count, 2)
            self.assertEqual(run_dict["best_feasible_metric"], 3.0)
            self.assertEqual(run_dict["best_feasible_stage"], "final")
            self.assertAlmostEqual(
                run_dict["best_feasible_incumbent"].search_eval["J_QS"],
                2.5e-4,
            )

    def test_maybe_update_best_hardware_near_miss_ranks_by_violation_first(self):
        module = load_single_stage_example_module()

        def hardware_status(violation_ratio):
            return {
                "success": False,
                "violations": ["poloidal_extent"],
                "violation_ratios": {"poloidal_extent": violation_ratio},
                "constraints": {},
            }

        run_dict = {
            "accepted_x": np.array([1.0, 2.0]),
            "surface_state": {"sdofs": [np.array([1.0])], "iota": [0.15], "G": [1.0]},
            "J": 8.0,
            "dJ": np.array([1.0, -1.0]),
            "search_eval": {"total": 8.0, "surface_weights": np.array([1.0])},
            "surface_status": {"success": True},
            "search_surface_status": {"success": True},
            "accepted_hardware_status": hardware_status(0.2),
            "topology_gate_status": {"enabled": False, "success": True},
            "intersecting": False,
            "best_hardware_near_miss_incumbent": None,
            "best_hardware_near_miss_metric": None,
            "best_hardware_near_miss_stage": None,
        }

        def rich_eval_for_current_state(surface_weights, *, include_diagnostics=None):
            self.assertTrue(include_diagnostics)
            np.testing.assert_allclose(
                surface_weights,
                run_dict["search_eval"]["surface_weights"],
            )
            return diagnostic_search_eval_payload(run_dict["search_eval"])

        with patch.object(
            module,
            "evaluate_search_objective",
            side_effect=rich_eval_for_current_state,
        ) as evaluate_mock:
            self.assertTrue(
                module.maybe_update_best_hardware_near_miss_incumbent(
                    run_dict,
                    "initial",
                    hardware_status=run_dict["accepted_hardware_status"],
                )
            )
            self.assertEqual(run_dict["best_hardware_near_miss_metric"], (0.2, 8.0))
            self.assertEqual(run_dict["best_hardware_near_miss_stage"], "initial")
            self.assertEqual(evaluate_mock.call_count, 1)

            run_dict["accepted_hardware_status"] = hardware_status(0.3)
            run_dict["search_eval"] = {"total": 1.0, "surface_weights": np.array([1.0])}
            run_dict["J"] = 1.0
            self.assertFalse(
                module.maybe_update_best_hardware_near_miss_incumbent(
                    run_dict,
                    "worse_hardware",
                    hardware_status=run_dict["accepted_hardware_status"],
                )
            )
            self.assertEqual(run_dict["best_hardware_near_miss_metric"], (0.2, 8.0))
            self.assertEqual(run_dict["best_hardware_near_miss_stage"], "initial")

            run_dict["accepted_hardware_status"] = hardware_status(0.1)
            run_dict["search_eval"] = {
                "total": 20.0,
                "surface_weights": np.array([1.0]),
            }
            run_dict["J"] = 20.0
            self.assertTrue(
                module.maybe_update_best_hardware_near_miss_incumbent(
                    run_dict,
                    "better_hardware",
                    hardware_status=run_dict["accepted_hardware_status"],
                )
            )
            self.assertEqual(run_dict["best_hardware_near_miss_metric"], (0.1, 20.0))
            self.assertEqual(
                run_dict["best_hardware_near_miss_stage"], "better_hardware"
            )
            self.assertEqual(evaluate_mock.call_count, 2)
            self.assertTrue(
                module.current_state_matches_best_hardware_near_miss_incumbent(
                    run_dict,
                    hardware_status=run_dict["accepted_hardware_status"],
                )
            )
            self.assertFalse(
                module.current_state_matches_best_hardware_near_miss_incumbent(
                    run_dict,
                    hardware_status={"success": True, "violations": []},
                )
            )

            run_dict["accepted_hardware_status"] = hardware_status(0.05)
            run_dict["search_eval"] = {"total": 0.5, "surface_weights": np.array([1.0])}
            run_dict["J"] = 0.5
            self.assertFalse(
                module.maybe_update_best_hardware_near_miss_incumbent(
                    run_dict,
                    "artifact_clean",
                    hardware_status={"success": True, "violations": []},
                )
            )
            self.assertEqual(run_dict["best_hardware_near_miss_metric"], (0.1, 20.0))
            self.assertEqual(
                run_dict["best_hardware_near_miss_stage"], "better_hardware"
            )

    def test_solver_checkpoint_preserves_best_hardware_near_miss_incumbent(self):
        module = load_single_stage_example_module()
        hardware_status = {
            "success": False,
            "violations": ["poloidal_extent"],
            "violation_ratios": {"poloidal_extent": 0.2},
            "constraints": {},
        }
        run_dict = {
            "accepted_x": np.array([1.0, 2.0]),
            "surface_state": {"sdofs": [np.array([1.0])], "iota": [0.15], "G": [1.0]},
            "J": 4.0,
            "dJ": np.array([1.0, -1.0]),
            "search_eval": {"total": 4.0, "surface_weights": np.array([1.0])},
            "surface_status": {"success": True},
            "search_surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"enabled": False, "success": True},
            "intersecting": False,
            "accepted_iterations": 3,
            "best_hardware_near_miss_incumbent": None,
            "best_hardware_near_miss_metric": None,
            "best_hardware_near_miss_stage": None,
        }

        def rich_eval_for_current_state(surface_weights, *, include_diagnostics=None):
            self.assertTrue(include_diagnostics)
            np.testing.assert_allclose(surface_weights, [1.0])
            return diagnostic_search_eval_payload(
                {"total": 4.0, "surface_weights": surface_weights}
            )

        with patch.object(
            module,
            "evaluate_search_objective",
            side_effect=rich_eval_for_current_state,
        ):
            self.assertTrue(
                module.maybe_update_best_hardware_near_miss_incumbent(
                    run_dict,
                    "initial",
                    hardware_status=hardware_status,
                )
            )
            payload = module.build_single_stage_solver_checkpoint_state(
                run_dict,
                requested_maxiter=10,
                runtime_maxiter=10,
                accepted_stage="initial",
                goal_mode="target",
                constraint_method="penalty",
                stage2_bs_path="/tmp/biot_savart.json",
                out_dir_iter="/tmp/out",
            )

        self.assertEqual(payload["best_hardware_near_miss_metric"], [0.2, 4.0])
        self.assertEqual(payload["best_hardware_near_miss_stage"], "initial")
        self.assertFalse(
            payload["best_hardware_near_miss_incumbent"]["accepted_hardware_status"][
                "success"
            ]
        )
        self.assertAlmostEqual(
            payload["best_hardware_near_miss_incumbent"]["search_eval"]["J_QS"],
            2.5e-4,
        )

    def test_solver_checkpoint_preserves_residue_objective_replay_config(self):
        module = load_single_stage_example_module()
        run_dict = {
            "accepted_x": np.array([1.0, 2.0]),
            "surface_state": {"sdofs": [np.array([1.0])], "iota": [0.15], "G": [1.0]},
            "J": 4.0,
            "dJ": np.array([1.0, -1.0]),
            "search_eval": diagnostic_search_eval_payload(
                {"total": 4.0, "surface_weights": np.array([1.0])}
            ),
            "surface_status": {"success": True},
            "search_surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"enabled": False, "success": True},
            "intersecting": False,
            "accepted_iterations": 3,
        }
        residue_replay_config = {
            "enabled": True,
            "weight": 0.25,
            "targets_json": "/tmp/targets.json",
            "targets_sha256": "sha256:targets",
            "seeds_json": "/tmp/seeds.json",
            "seeds_sha256": "sha256:seeds",
            "target_manifest_id": "sha256:test-targets",
            "validation_id": "validation-artifact",
            "axis_r": 0.86,
            "axis_z": 0.0,
        }

        payload = module.build_single_stage_solver_checkpoint_state(
            run_dict,
            requested_maxiter=10,
            runtime_maxiter=10,
            accepted_stage="initial",
            goal_mode="target",
            constraint_method="penalty",
            stage2_bs_path="/tmp/biot_savart.json",
            out_dir_iter="/tmp/out",
            residue_objective_replay_config=residue_replay_config,
        )

        self.assertEqual(
            payload["residue_objective_replay_config"],
            residue_replay_config,
        )

    def test_resume_solver_checkpoint_requires_matching_residue_replay_config(self):
        module = load_single_stage_example_module()
        checkpoint_payload = {
            "residue_objective_replay_config": {
                "enabled": True,
                "weight": 0.25,
                "targets_sha256": "sha256:targets",
                "seeds_sha256": "sha256:seeds",
                "target_manifest_id": "sha256:test-targets",
                "validation_id": "validation-artifact",
            }
        }
        matching_config = tuple(
            checkpoint_payload["residue_objective_replay_config"].items()
        )

        module.validate_resume_solver_checkpoint_residue_replay_config(
            checkpoint_payload,
            matching_config,
        )
        with self.assertRaisesRegex(
            ValueError, "Greene residue objective replay config"
        ):
            module.validate_resume_solver_checkpoint_residue_replay_config(
                checkpoint_payload,
                None,
            )
        drifted_config = dict(checkpoint_payload["residue_objective_replay_config"])
        drifted_config["target_manifest_id"] = "sha256:other-targets"
        with self.assertRaisesRegex(
            ValueError,
            "Greene residue objective replay config",
        ):
            module.validate_resume_solver_checkpoint_residue_replay_config(
                checkpoint_payload,
                tuple(drifted_config.items()),
            )

    def test_maybe_update_best_accepted_incumbent_tracks_valid_nonself_intersecting_states(
        self,
    ):
        module = load_single_stage_example_module()
        run_dict = {
            "accepted_x": np.array([1.0, 2.0]),
            "surface_state": {"sdofs": [np.array([1.0])], "iota": [0.15], "G": [1.0]},
            "J": 4.0,
            "dJ": np.array([1.0, -1.0]),
            "search_eval": {"total": 4.0, "surface_weights": np.array([1.0])},
            "surface_status": {"success": True},
            "search_surface_status": {"success": True},
            "accepted_hardware_status": {
                "success": False,
                "violations": ["max_curvature"],
            },
            "topology_gate_status": {"enabled": False, "success": True},
            "intersecting": False,
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
        }

        def rich_eval_for_current_state(surface_weights, *, include_diagnostics=None):
            self.assertTrue(include_diagnostics)
            np.testing.assert_allclose(
                surface_weights,
                run_dict["search_eval"]["surface_weights"],
            )
            return diagnostic_search_eval_payload(run_dict["search_eval"])

        with patch.object(
            module,
            "evaluate_search_objective",
            side_effect=rich_eval_for_current_state,
        ) as evaluate_mock:
            self.assertTrue(
                module.maybe_update_best_accepted_incumbent(run_dict, "initial")
            )
            self.assertEqual(evaluate_mock.call_count, 1)
            self.assertEqual(run_dict["best_accepted_metric"], 4.0)
            self.assertEqual(run_dict["best_accepted_stage"], "initial")
            np.testing.assert_allclose(
                run_dict["best_accepted_incumbent"].x,
                [1.0, 2.0],
            )
            self.assertNotIn("J_QS", run_dict["search_eval"])
            self.assertAlmostEqual(
                run_dict["best_accepted_incumbent"].search_eval["J_QS"],
                2.5e-4,
            )

            run_dict["search_eval"] = {
                "total": 5.0,
                "surface_weights": np.array([1.0]),
            }
            run_dict["J"] = 5.0
            self.assertFalse(
                module.maybe_update_best_accepted_incumbent(run_dict, "middle")
            )
            self.assertEqual(evaluate_mock.call_count, 1)
            self.assertEqual(run_dict["best_accepted_metric"], 4.0)

            run_dict["intersecting"] = True
            run_dict["search_eval"] = {
                "total": 3.0,
                "surface_weights": np.array([1.0]),
            }
            run_dict["J"] = 3.0
            self.assertFalse(
                module.maybe_update_best_accepted_incumbent(run_dict, "final")
            )
            self.assertEqual(evaluate_mock.call_count, 1)
            self.assertEqual(run_dict["best_accepted_metric"], 4.0)

    def test_solver_checkpoint_accepted_incumbent_recomputes_compact_diagnostics(self):
        module = load_single_stage_example_module()
        run_dict = {
            "accepted_x": np.array([1.0, 2.0]),
            "surface_state": {"sdofs": [np.array([1.0])], "iota": [0.15], "G": [1.0]},
            "J": 4.0,
            "dJ": np.array([1.0, -1.0]),
            "search_eval": {"total": 4.0, "surface_weights": np.array([1.0])},
            "surface_status": {"success": True},
            "search_surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"enabled": False, "success": True},
            "accepted_iterations": 2,
        }

        def rich_eval_for_current_state(surface_weights, *, include_diagnostics=None):
            self.assertTrue(include_diagnostics)
            np.testing.assert_allclose(
                surface_weights,
                run_dict["search_eval"]["surface_weights"],
            )
            return diagnostic_search_eval_payload(run_dict["search_eval"])

        with patch.object(
            module,
            "evaluate_search_objective",
            side_effect=rich_eval_for_current_state,
        ) as evaluate_mock:
            payload = module.build_single_stage_solver_checkpoint_state(
                run_dict,
                requested_maxiter=10,
                runtime_maxiter=10,
                accepted_stage="initial",
                goal_mode="target",
                constraint_method="penalty",
                stage2_bs_path="/tmp/biot_savart.json",
                out_dir_iter="/tmp/out",
            )

        self.assertEqual(evaluate_mock.call_count, 1)
        self.assertNotIn("J_QS", run_dict["search_eval"])
        self.assertAlmostEqual(
            payload["accepted_incumbent"]["search_eval"]["J_QS"],
            2.5e-4,
        )

    def test_resume_incumbent_normalization_recomputes_legacy_compact_best_state(self):
        module = load_single_stage_example_module()
        compact_best_state = {
            "accepted_x": np.array([1.0, 2.0]),
            "surface_state": {"sdofs": [np.array([1.0])], "iota": [0.15], "G": [1.0]},
            "J": 4.0,
            "dJ": np.array([1.0, -1.0]),
            "search_eval": {"total": 4.0, "surface_weights": np.array([1.0])},
            "surface_status": {"success": True},
            "search_surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"enabled": False, "success": True},
        }
        current_state = {
            "accepted_x": np.array([9.0, 8.0]),
            "surface_state": {"sdofs": [np.array([9.0])], "iota": [0.16], "G": [1.0]},
            "J": 6.0,
            "dJ": np.array([2.0, -2.0]),
            "search_eval": diagnostic_search_eval_payload(
                {"total": 6.0, "surface_weights": np.array([1.0])}
            ),
            "surface_status": {"success": True},
            "search_surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"enabled": False, "success": True},
            "accepted_boozer_stage": "current_stage",
        }
        compact_incumbent = module.snapshot_single_stage_incumbent_state(
            compact_best_state
        )
        rebuilt_stages = []

        def rebuild_stage(stage_name):
            rebuilt_stages.append(stage_name)
            return {}

        def refresh_state(run_state, accepted_stage):
            run_state["accepted_boozer_stage"] = accepted_stage
            return float(run_state["search_eval"]["total"])

        def rich_eval_for_current_state(surface_weights, *, include_diagnostics=None):
            self.assertTrue(include_diagnostics)
            np.testing.assert_allclose(surface_weights, [1.0])
            return diagnostic_search_eval_payload(
                {"total": 4.0, "surface_weights": surface_weights}
            )

        with (
            patch.object(
                module,
                "refresh_accepted_search_state",
                side_effect=refresh_state,
            ),
            patch.object(
                module,
                "evaluate_search_objective",
                side_effect=rich_eval_for_current_state,
            ) as evaluate_mock,
        ):
            normalized = module.normalize_diagnostic_incumbent_for_stage(
                current_state,
                compact_incumbent,
                "best_stage",
                "current_stage",
                rebuild_stage,
            )

        self.assertEqual(evaluate_mock.call_count, 1)
        self.assertEqual(rebuilt_stages, ["best_stage", "current_stage"])
        self.assertAlmostEqual(normalized.search_eval["J_QS"], 2.5e-4)
        np.testing.assert_allclose(current_state["accepted_x"], [9.0, 8.0])
        self.assertEqual(current_state["accepted_boozer_stage"], "current_stage")

    def test_frontier_preserved_incumbents_require_trust_and_frontier_rank_metric(self):
        module = load_single_stage_example_module()
        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        run_dict = {
            "accepted_x": np.array([1.0, 2.0]),
            "surface_state": {"sdofs": [np.array([1.0])], "iota": [0.15], "G": [1.0]},
            "J": 4.0,
            "dJ": np.array([1.0, -1.0]),
            "search_eval": diagnostic_search_eval_payload(
                {
                    "total": 100.0,
                    "frontier_rank_total": 4.0,
                    "frontier_trust_ok": True,
                    "surface_weights": np.array([1.0]),
                }
            ),
            "surface_status": {"success": True},
            "search_surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"enabled": False, "success": True},
            "intersecting": False,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
        }
        module.refresh_frontier_certification_status(
            run_dict,
            hardware_status=run_dict["accepted_hardware_status"],
            accepted_iteration=0,
            topology_entry={
                "accepted_iteration": 0,
                "topology_broken": False,
                "invariant_torus_fraction": 0.5,
                "kam_fraction": 0.5,
            },
        )

        self.assertTrue(
            module.maybe_update_best_feasible_incumbent(run_dict, "initial")
        )
        self.assertEqual(run_dict["best_feasible_metric"], 4.0)

        run_dict["search_eval"]["frontier_trust_ok"] = False
        run_dict["search_eval"]["frontier_rank_total"] = 3.0
        run_dict["J"] = 3.0
        self.assertFalse(module.maybe_update_best_feasible_incumbent(run_dict, "final"))
        self.assertEqual(run_dict["best_feasible_metric"], 4.0)

        run_dict["search_eval"]["frontier_trust_ok"] = True
        run_dict["search_eval"]["finite_eval_ok"] = False
        run_dict["search_eval"]["frontier_rank_total"] = 2.0
        run_dict["J"] = 2.0
        self.assertFalse(module.maybe_update_best_feasible_incumbent(run_dict, "final"))
        self.assertEqual(run_dict["best_feasible_metric"], 4.0)

    def test_frontier_refinement_labels_uncertified_without_starving_best_feasible(
        self,
    ):
        module = load_single_stage_example_module()
        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_INVARIANT_TORUS_MIN = 0.30
        module.FRONTIER_KAM_MIN = 0.30
        run_dict = {
            "accepted_x": np.array([1.0, 2.0]),
            "surface_state": {"sdofs": [np.array([1.0])], "iota": [0.15], "G": [1.0]},
            "J": 4.0,
            "dJ": np.array([1.0, -1.0]),
            "search_eval": diagnostic_search_eval_payload(
                {
                    "total": 4.0,
                    "frontier_rank_total": 4.0,
                    "frontier_trust_ok": True,
                    "surface_weights": np.array([1.0]),
                }
            ),
            "surface_status": {"success": True},
            "search_surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"enabled": True, "success": True},
            "intersecting": False,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
        }

        low_kam_status = module.refresh_frontier_certification_status(
            run_dict,
            hardware_status=run_dict["accepted_hardware_status"],
            accepted_iteration=5,
            topology_entry={
                "accepted_iteration": 5,
                "topology_broken": False,
                "invariant_torus_fraction": 1.0 / 12.0,
                "kam_fraction": 1.0 / 12.0,
            },
        )

        self.assertFalse(low_kam_status["ok"])
        self.assertEqual(
            low_kam_status["reason"],
            "invariant_torus_fraction_below_min",
        )
        self.assertTrue(module.refinement_eligible_incumbent(run_dict))
        self.assertTrue(
            module.maybe_update_best_feasible_incumbent(run_dict, "accepted")
        )
        self.assertFalse(run_dict["search_eval"]["frontier_certification_ok"])
        self.assertAlmostEqual(
            run_dict["search_eval"]["frontier_invariant_torus_fraction"],
            1.0 / 12.0,
        )
        self.assertAlmostEqual(
            run_dict["search_eval"]["frontier_kam_fraction"], 1.0 / 12.0
        )
        self.assertFalse(
            run_dict["best_feasible_incumbent"].search_eval["frontier_certification_ok"]
        )
        self.assertFalse(module.frontier_reportable_success(True, True, low_kam_status))
        self.assertIsNone(
            module.frontier_reportable_success(False, True, low_kam_status)
        )

        run_dict["J"] = 3.0
        run_dict["search_eval"] = diagnostic_search_eval_payload(
            {
                "total": 3.0,
                "frontier_rank_total": 3.0,
                "frontier_trust_ok": True,
                "surface_weights": np.array([1.0]),
            }
        )
        high_kam_status = module.refresh_frontier_certification_status(
            run_dict,
            hardware_status=run_dict["accepted_hardware_status"],
            accepted_iteration=6,
            topology_entry={
                "accepted_iteration": 6,
                "topology_broken": False,
                "invariant_torus_fraction": 0.5,
                "kam_fraction": 0.5,
            },
        )
        self.assertTrue(high_kam_status["ok"])
        self.assertEqual(high_kam_status["reason"], "certified")
        self.assertTrue(module.refinement_eligible_incumbent(run_dict))
        self.assertTrue(
            module.maybe_update_best_feasible_incumbent(run_dict, "certified")
        )
        self.assertEqual(run_dict["best_feasible_stage"], "certified")
        self.assertTrue(
            run_dict["best_feasible_incumbent"].search_eval["frontier_certification_ok"]
        )
        self.assertTrue(module.frontier_reportable_success(True, True, high_kam_status))

    def test_frontier_best_feasible_remains_raw_metric_ordered(self):
        module = load_single_stage_example_module()
        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_INVARIANT_TORUS_MIN = 0.30
        module.FRONTIER_KAM_MIN = 0.30
        run_dict = {
            "accepted_x": np.array([1.0, 2.0]),
            "surface_state": {"sdofs": [np.array([1.0])], "iota": [0.15], "G": [1.0]},
            "J": 1.0,
            "dJ": np.array([1.0, -1.0]),
            "search_eval": diagnostic_search_eval_payload(
                {
                    "total": 1.0,
                    "frontier_rank_total": 1.0,
                    "frontier_trust_ok": True,
                    "surface_weights": np.array([1.0]),
                }
            ),
            "surface_status": {"success": True},
            "search_surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"enabled": True, "success": True},
            "intersecting": False,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
        }
        module.refresh_frontier_certification_status(
            run_dict,
            hardware_status=run_dict["accepted_hardware_status"],
            accepted_iteration=1,
            topology_entry={
                "accepted_iteration": 1,
                "topology_broken": False,
                "invariant_torus_fraction": 1.0 / 12.0,
                "kam_fraction": 1.0 / 12.0,
            },
        )
        self.assertTrue(
            module.maybe_update_best_feasible_incumbent(run_dict, "raw_low")
        )
        self.assertEqual(run_dict["best_feasible_metric"], 1.0)
        self.assertEqual(run_dict["best_feasible_stage"], "raw_low")
        self.assertFalse(
            run_dict["best_feasible_incumbent"].search_eval["frontier_certification_ok"]
        )

        run_dict["accepted_x"] = np.array([3.0, 4.0])
        run_dict["J"] = 100.0
        run_dict["search_eval"] = diagnostic_search_eval_payload(
            {
                "total": 100.0,
                "frontier_rank_total": 100.0,
                "frontier_trust_ok": True,
                "surface_weights": np.array([1.0]),
            }
        )
        module.refresh_frontier_certification_status(
            run_dict,
            hardware_status=run_dict["accepted_hardware_status"],
            accepted_iteration=2,
            topology_entry={
                "accepted_iteration": 2,
                "topology_broken": False,
                "invariant_torus_fraction": 0.5,
                "kam_fraction": 0.5,
            },
        )

        self.assertFalse(
            module.maybe_update_best_feasible_incumbent(run_dict, "certified_worse")
        )
        self.assertEqual(run_dict["best_feasible_metric"], 1.0)
        self.assertEqual(run_dict["best_feasible_stage"], "raw_low")
        np.testing.assert_allclose(run_dict["best_feasible_incumbent"].x, [1.0, 2.0])

    def test_frontier_explicit_zero_kam_floor_reports_hardware_failed_not_kam_deficit(
        self,
    ):
        module = load_single_stage_example_module()
        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_INVARIANT_TORUS_MIN = 0.0
        module.FRONTIER_KAM_MIN = 0.0
        run_dict = {
            "search_eval": {
                "total": 4.0,
                "frontier_rank_total": 4.0,
            },
        }

        status = module.refresh_frontier_certification_status(
            run_dict,
            hardware_status={"success": False, "violations": ["hardware failed"]},
            accepted_iteration=9,
            topology_entry={
                "accepted_iteration": 9,
                "topology_broken": False,
                "invariant_torus_fraction": 1.0 / 12.0,
                "kam_fraction": 1.0 / 12.0,
            },
        )

        self.assertFalse(status["ok"])
        self.assertEqual(status["reason"], "hardware_failed")
        self.assertFalse(status["hardware_ok"])
        self.assertAlmostEqual(status["invariant_torus_fraction"], 1.0 / 12.0)
        self.assertEqual(status["invariant_torus_min"], 0.0)
        self.assertEqual(status["invariant_torus_deficit"], 0.0)
        self.assertAlmostEqual(status["kam_fraction"], 1.0 / 12.0)
        self.assertEqual(status["kam_min"], 0.0)
        self.assertEqual(status["kam_deficit"], 0.0)
        self.assertFalse(module.frontier_reportable_success(True, True, status))
        self.assertIsNone(module.frontier_reportable_success(False, True, status))
        self.assertFalse(run_dict["search_eval"]["frontier_certification_ok"])
        self.assertEqual(
            run_dict["search_eval"]["frontier_certification_reason"],
            "hardware_failed",
        )

    def test_frontier_default_kam_floor_is_nonzero_and_rejects_low_wba_fraction(self):
        module = load_single_stage_example_module()
        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_INVARIANT_TORUS_MIN = (
            module._default_frontier_invariant_torus_min_impl()
        )
        module.FRONTIER_KAM_MIN = module.FRONTIER_INVARIANT_TORUS_MIN
        run_dict = {"search_eval": {"total": 4.0, "frontier_rank_total": 4.0}}

        status = module.refresh_frontier_certification_status(
            run_dict,
            hardware_status={"success": True, "violations": []},
            accepted_iteration=9,
            topology_entry={
                "accepted_iteration": 9,
                "topology_broken": False,
                "invariant_torus_fraction": 1.0 / 12.0,
                "kam_fraction": 1.0 / 12.0,
                "kam_fraction_semantics": module.KAM_FRACTION_SEMANTICS,
            },
        )

        self.assertFalse(status["ok"])
        self.assertEqual(status["reason"], "invariant_torus_fraction_below_min")
        self.assertEqual(status["invariant_torus_min"], 0.30)
        self.assertGreater(status["invariant_torus_deficit"], 0.0)

    def test_frontier_certification_requires_invariant_torus_semantics_for_legacy_kam_field(
        self,
    ):
        module = load_single_stage_example_module()

        legacy_proxy_status = module._evaluate_frontier_kam_certification_impl(
            {
                "accepted_iteration": 1,
                "topology_broken": False,
                "kam_fraction": 0.5,
            },
            enabled=True,
            hardware_ok=True,
            kam_min=0.30,
            accepted_iteration=1,
        )
        wba_alias_status = module._evaluate_frontier_kam_certification_impl(
            {
                "accepted_iteration": 1,
                "topology_broken": False,
                "kam_fraction": 0.5,
                "kam_fraction_semantics": KAM_FRACTION_SEMANTICS,
            },
            enabled=True,
            hardware_ok=True,
            kam_min=0.30,
            accepted_iteration=1,
        )

        self.assertFalse(legacy_proxy_status["ok"])
        self.assertEqual(
            legacy_proxy_status["reason"],
            "invariant_torus_fraction_missing",
        )
        self.assertTrue(wba_alias_status["ok"])
        self.assertEqual(wba_alias_status["reason"], "certified")

    def test_resume_preserves_current_frontier_certification_status(self):
        module = load_single_stage_example_module()
        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        run_dict = {
            "search_eval": {
                "total": 4.0,
                "frontier_rank_total": 4.0,
            },
        }
        stored_certification_status = {
            "enabled": True,
            "ok": True,
            "reason": "certified",
            "hardware_ok": True,
            "topology_evaluated": True,
            "topology_broken": False,
            "accepted_iteration": 5,
            "topology_accepted_iteration": 5,
            "invariant_torus_fraction": 0.5,
            "invariant_torus_min": 0.30,
            "invariant_torus_deficit": 0.0,
            "kam_fraction": 0.5,
            "kam_min": 0.30,
            "kam_deficit": 0.0,
        }

        restored = module.restore_or_refresh_frontier_certification_status_after_resume(
            run_dict,
            stored_certification_status=stored_certification_status,
            hardware_status={"success": True, "violations": []},
            accepted_iteration=5,
        )

        self.assertEqual(restored["reason"], "certified")
        self.assertTrue(run_dict["search_eval"]["frontier_certification_ok"])
        self.assertEqual(
            run_dict["search_eval"]["frontier_certification_reason"], "certified"
        )
        self.assertAlmostEqual(
            run_dict["search_eval"]["frontier_invariant_torus_fraction"],
            0.5,
        )
        self.assertAlmostEqual(run_dict["search_eval"]["frontier_kam_fraction"], 0.5)

    def test_resume_rejects_legacy_kam_only_frontier_certification_status(self):
        module = load_single_stage_example_module()
        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        run_dict = {
            "search_eval": {
                "total": 4.0,
                "frontier_rank_total": 4.0,
            },
        }
        stored_certification_status = {
            "enabled": True,
            "ok": True,
            "reason": "certified",
            "hardware_ok": True,
            "topology_evaluated": True,
            "topology_broken": False,
            "accepted_iteration": 5,
            "topology_accepted_iteration": 5,
            "kam_fraction": 0.5,
            "kam_min": 0.30,
            "kam_deficit": 0.0,
        }

        restored = module.restore_or_refresh_frontier_certification_status_after_resume(
            run_dict,
            stored_certification_status=stored_certification_status,
            hardware_status={"success": True, "violations": []},
            accepted_iteration=5,
        )

        self.assertFalse(restored["ok"])
        self.assertEqual(restored["reason"], "topology_not_evaluated")
        self.assertFalse(run_dict["search_eval"]["frontier_certification_ok"])
        self.assertEqual(
            run_dict["search_eval"]["frontier_certification_reason"],
            "topology_not_evaluated",
        )

    def test_topology_survival_rank_key_ignores_unstamped_legacy_kam_fraction(self):
        module = load_single_stage_example_module()

        legacy_proxy = {
            "survival_fraction": 1.0,
            "kam_fraction": 1.0,
            "confinement_score": 1.0,
        }
        wba_current = {
            "survival_fraction": 1.0,
            "invariant_torus_fraction": 0.2,
            "confinement_score": 0.0,
        }

        self.assertLess(
            module.topology_survival_rank_key(legacy_proxy),
            module.topology_survival_rank_key(wba_current),
        )

    def test_refinement_eligible_incumbent_requires_topology_success(self):
        module = load_single_stage_example_module()
        run_dict = {
            "accepted_hardware_status": {"success": True},
            "topology_gate_status": {"enabled": True, "success": False},
            "surface_status": {"success": True},
            "search_eval": {"total": 1.0},
            "intersecting": False,
        }

        self.assertFalse(module.refinement_eligible_incumbent(run_dict))

        run_dict["topology_gate_status"] = {"enabled": True, "success": True}
        self.assertTrue(module.refinement_eligible_incumbent(run_dict))

    def test_write_preserved_timeout_artifacts_uses_kind_specific_filenames(self):
        module = load_single_stage_example_module()

        class FakeBiotSavart:
            def __init__(self):
                self.saved = []

            def save(self, path):
                self.saved.append(path)

        class FakeSurface:
            def __init__(self):
                self.saved = []

            def save(self, path):
                self.saved.append(path)

        class FakeBoozerSurface:
            def __init__(self):
                self.surface = FakeSurface()
                self.saved = []

            def save(self, path):
                self.saved.append(path)

        fake_bs = FakeBiotSavart()
        fake_outer = FakeBoozerSurface()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            payload = {"FIELD_ERROR": 0.01}
            module.write_preserved_timeout_artifacts(
                out_dir,
                preservation_kind="best_feasible",
                results_payload=payload,
                biotsavart=fake_bs,
                surface_data=[{"name": "outer", "boozer_surface": fake_outer}],
            )

            partial_results = out_dir / "results_best_feasible.partial.json"
            self.assertTrue(partial_results.exists())
            self.assertEqual(
                json.loads(partial_results.read_text(encoding="utf-8")), payload
            )
            self.assertEqual(
                fake_bs.saved, [str(out_dir / "biot_savart_best_feasible.json")]
            )
            self.assertEqual(
                fake_outer.surface.saved,
                [str(out_dir / "surf_best_feasible_outer.json")],
            )
            self.assertEqual(
                fake_outer.saved,
                [str(out_dir / "surf_best_feasible_outer_boozer_surface.json")],
            )
            module.write_preserved_timeout_artifacts(
                out_dir,
                preservation_kind="best_hardware_near_miss",
                results_payload=payload,
                biotsavart=fake_bs,
                surface_data=[{"name": "outer", "boozer_surface": fake_outer}],
            )

            near_miss_results = out_dir / "results_best_hardware_near_miss.partial.json"
            self.assertTrue(near_miss_results.exists())
            self.assertEqual(
                fake_bs.saved[-1],
                str(out_dir / "biot_savart_best_hardware_near_miss.json"),
            )
            self.assertEqual(
                fake_outer.surface.saved[-1],
                str(out_dir / "surf_best_hardware_near_miss_outer.json"),
            )
            self.assertEqual(
                fake_outer.saved[-1],
                str(out_dir / "surf_best_hardware_near_miss_outer_boozer_surface.json"),
            )

    def test_write_preserved_timeout_artifacts_marks_strict_vacuum_sidecar(self):
        module = load_single_stage_example_module()

        class FakeBiotSavart:
            def save(self, path):
                Path(path).write_text('{"coils": []}', encoding="utf-8")

        class FakeSurface:
            def save(self, path):
                Path(path).write_text('{"surface": "outer"}', encoding="utf-8")

        class FakeBoozerSurface:
            def __init__(self):
                self.surface = FakeSurface()
                self.res = {"iota": 0.3588536981, "G": -0.377}

            def save(self, path):
                Path(path).write_text(
                    json.dumps(
                        {
                            "simsopt_objs": {
                                "surface": {
                                    "@module": "simsopt.geo.boozersurface",
                                    "@class": "BoozerSurface",
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            module.write_preserved_timeout_artifacts(
                out_dir,
                preservation_kind="best_hardware_near_miss",
                results_payload={
                    "STRICT_VACUUM_CURRENT": True,
                    "CURRENT_LINEAGE": "strict_vacuum",
                    "STRICT_VACUUM_SEED_LINEAGE": "legacy_control",
                    "TF_CURRENT_A": -80000.0,
                    "BANANA_CURRENTS_A": [-15910.0],
                    "EFFECTIVE_CURRENT_MODE": "vacuum",
                    "FINITE_CURRENT_MODE": None,
                    "PLASMA_CURRENT_A": 0.0,
                    "BOOZER_I": 0.0,
                    "PROXY_PLASMA_CURRENT_A": 0.0,
                    "VF_CURRENT_A": 0.0,
                    "NUM_PROXY_COILS": 0,
                    "NUM_VF_COILS": 0,
                    "BOOZER_SURFACE_CLASS": "BoozerSurface",
                    "BOOZER_SURFACE_MODULE": "simsopt.geo.boozersurface",
                },
                biotsavart=FakeBiotSavart(),
                surface_data=[{"name": "outer", "boozer_surface": FakeBoozerSurface()}],
            )

            state_payload = json.loads(
                (
                    out_dir / "surf_best_hardware_near_miss_outer_boozer_state.json"
                ).read_text(encoding="utf-8")
            )

        manifest = state_payload["interchange_manifest"]
        self.assertAlmostEqual(state_payload["iota"], 0.3588536981)
        self.assertAlmostEqual(state_payload["G"], -0.377)
        self.assertEqual(manifest["current_lineage"], "strict_vacuum")
        self.assertTrue(manifest["baseline_replayable"])
        self.assertEqual(
            manifest["requires_boozer_surface_module"],
            "simsopt.geo.boozersurface",
        )
        self.assertEqual(manifest["requires_boozer_surface_class"], "BoozerSurface")
        self.assertTrue(manifest["requires_no_boozer_surface_i_field"])
        self.assertFalse(manifest["boozer_surface_has_i_field"])

    def test_write_preserved_timeout_artifacts_requires_strict_vacuum_sidecar(self):
        module = load_single_stage_example_module()

        class FakeBiotSavart:
            def save(self, path):
                Path(path).write_text('{"coils": []}', encoding="utf-8")

        class FakeSurface:
            def save(self, path):
                Path(path).write_text('{"surface": "outer"}', encoding="utf-8")

        class UnsolvedBoozerSurface:
            def __init__(self):
                self.surface = FakeSurface()
                self.res = {}

            def save(self, path):
                Path(path).write_text(
                    json.dumps(
                        {
                            "simsopt_objs": {
                                "surface": {
                                    "@module": "simsopt.geo.boozersurface",
                                    "@class": "BoozerSurface",
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with self.assertRaisesRegex(ValueError, "solved iota/G state sidecar"):
                module.write_preserved_timeout_artifacts(
                    out_dir,
                    preservation_kind="best_hardware_near_miss",
                    results_payload={
                        "STRICT_VACUUM_CURRENT": True,
                        "CURRENT_LINEAGE": "strict_vacuum",
                        "STRICT_VACUUM_SEED_LINEAGE": "legacy_control",
                        "TF_CURRENT_A": -80000.0,
                        "BANANA_CURRENTS_A": [-15910.0],
                        "EFFECTIVE_CURRENT_MODE": "vacuum",
                        "FINITE_CURRENT_MODE": None,
                        "PLASMA_CURRENT_A": 0.0,
                        "BOOZER_I": 0.0,
                        "PROXY_PLASMA_CURRENT_A": 0.0,
                        "VF_CURRENT_A": 0.0,
                        "NUM_PROXY_COILS": 0,
                        "NUM_VF_COILS": 0,
                        "BOOZER_SURFACE_CLASS": "BoozerSurface",
                        "BOOZER_SURFACE_MODULE": "simsopt.geo.boozersurface",
                    },
                    biotsavart=FakeBiotSavart(),
                    surface_data=[
                        {"name": "outer", "boozer_surface": UnsolvedBoozerSurface()}
                    ],
                )
            self.assertFalse(
                (
                    out_dir / "surf_best_hardware_near_miss_outer_boozer_state.json"
                ).exists()
            )

    def test_write_preserved_timeout_artifacts_rejects_strict_vacuum_finite_i(self):
        module = load_single_stage_example_module()

        class FakeBiotSavart:
            def save(self, path):
                Path(path).write_text('{"coils": []}', encoding="utf-8")

        class FakeSurface:
            def save(self, path):
                Path(path).write_text('{"surface": "outer"}', encoding="utf-8")

        class FiniteIBoozerSurface:
            def __init__(self):
                self.surface = FakeSurface()
                self.res = {"iota": 0.12, "G": -0.377}
                self.I = 1.0e-3

            def save(self, path):
                Path(path).write_text(
                    json.dumps(
                        {
                            "simsopt_objs": {
                                "surface": {
                                    "@module": "banana_opt.boozer_finite_current",
                                    "@class": "BoozerSurfaceFiniteI",
                                    "I": 1.0e-3,
                                }
                            }
                        }
                    ),
                    encoding="utf-8",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with self.assertRaisesRegex(ValueError, "without an I field"):
                module.write_preserved_timeout_artifacts(
                    out_dir,
                    preservation_kind="best_hardware_near_miss",
                    results_payload={
                        "STRICT_VACUUM_CURRENT": True,
                        "CURRENT_LINEAGE": "strict_vacuum",
                        "STRICT_VACUUM_SEED_LINEAGE": "legacy_control",
                        "TF_CURRENT_A": -80000.0,
                        "BANANA_CURRENTS_A": [-15910.0],
                        "EFFECTIVE_CURRENT_MODE": "vacuum",
                        "FINITE_CURRENT_MODE": None,
                        "PLASMA_CURRENT_A": 0.0,
                        "BOOZER_I": 0.0,
                        "PROXY_PLASMA_CURRENT_A": 0.0,
                        "VF_CURRENT_A": 0.0,
                        "NUM_PROXY_COILS": 0,
                        "NUM_VF_COILS": 0,
                        "BOOZER_SURFACE_CLASS": "BoozerSurface",
                        "BOOZER_SURFACE_MODULE": "simsopt.geo.boozersurface",
                    },
                    biotsavart=FakeBiotSavart(),
                    surface_data=[
                        {"name": "outer", "boozer_surface": FiniteIBoozerSurface()}
                    ],
                )
            self.assertFalse(
                (
                    out_dir / "surf_best_hardware_near_miss_outer_boozer_state.json"
                ).exists()
            )

    def test_write_preserved_timeout_artifacts_rejects_strict_positive_currents(self):
        module = load_single_stage_example_module()

        class FakeBiotSavart:
            def save(self, path):
                Path(path).write_text('{"coils": []}', encoding="utf-8")

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            self.assertRaisesRegex(ValueError, "signed_tf_current_negative"),
        ):
            module.write_preserved_timeout_artifacts(
                Path(tmpdir),
                preservation_kind="best_hardware_near_miss",
                results_payload={
                    "STRICT_VACUUM_CURRENT": True,
                    "CURRENT_LINEAGE": "strict_vacuum",
                    "STRICT_VACUUM_SEED_LINEAGE": "legacy_control",
                    "TF_CURRENT_A": 80000.0,
                    "BANANA_CURRENT_A": 15910.0,
                    "EFFECTIVE_CURRENT_MODE": "vacuum",
                    "FINITE_CURRENT_MODE": None,
                    "PLASMA_CURRENT_A": 0.0,
                    "BOOZER_I": 0.0,
                    "PROXY_PLASMA_CURRENT_A": 0.0,
                    "VF_CURRENT_A": 0.0,
                    "NUM_PROXY_COILS": 0,
                    "NUM_VF_COILS": 0,
                    "BOOZER_SURFACE_CLASS": "BoozerSurface",
                    "BOOZER_SURFACE_MODULE": "simsopt.geo.boozersurface",
                },
                biotsavart=FakeBiotSavart(),
                surface_data=[{"name": "outer", "boozer_surface": object()}],
            )

    def test_preserved_timeout_artifacts_recompute_diagnostics_for_compact_search_eval(
        self,
    ):
        module = load_single_stage_example_module()

        class FakeBiotSavart:
            def __init__(self):
                self.saved = []

            def save(self, path):
                self.saved.append(path)

        class FakeSurface:
            def save(self, path):
                self.saved = path

        class FakeBoozerSurface:
            def __init__(self):
                self.surface = FakeSurface()

            def save(self, path):
                self.saved = path

        fake_bs = FakeBiotSavart()
        fake_outer = FakeBoozerSurface()
        module.PRESERVED_TIMEOUT_REPLAY_CONFIG = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_test.nc",
            plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
            toroidal_flux=0.24,
            banana_surf_radius=0.142,
            curvature_threshold=100.0,
            order=2,
            stage2_seed_surf_path="/seeds/surf_opt_boozer_surface.json",
            single_stage_goal_mode="target",
        )
        compact_eval = {
            "total": 7.5e-4,
            "base_total": 7.4e-4,
            "grad": np.array([1.0]),
            "surface_weights": np.array([1.0]),
            "diagnostics_included": False,
        }
        run_dict = {
            "search_eval": compact_eval,
            "J": 7.5e-4,
            "intersecting": False,
            "accepted_iterations": 0,
            "surface_status": {
                "success": True,
                "iotas": [0.14997],
                "volumes": [0.09998],
            },
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"success": True},
        }
        hardware_snapshot = {
            "search_hardware_status": {"success": True, "violations": []},
            "artifact_hardware_status": {"success": True, "violations": []},
            "max_curvature": 19.8,
            "length_target": 1.7,
            "tf_current_A": -8.0e4,
            "tf_current_limit_A": 8.0e4,
            "banana_current_A": 1.4e4,
            "banana_current_max_A": 1.6e4,
            "curve_curve_min_dist": 0.0496,
            "curve_surface_min_dist": 0.067,
            "surface_vessel_min_dist": 0.082,
        }
        rich_eval = dict(compact_eval, J_QS=2.7e-4, J_Boozer=4.8e-7)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(
                module,
                "evaluate_search_objective",
                return_value=rich_eval,
            ) as evaluate_mock,
        ):
            out_dir = Path(tmpdir)
            module.write_preserved_timeout_artifacts_for_current_state(
                out_dir,
                preservation_kind="best_accepted",
                incumbent_stage="initial",
                run_dict=run_dict,
                bs=fake_bs,
                surface_data=[{"name": "outer", "boozer_surface": fake_outer}],
                hardware_snapshot=hardware_snapshot,
                field_error=3.5e-4,
                coil_length=2.91,
            )
            payload = json.loads(
                (out_dir / "results_best_accepted.partial.json").read_text(
                    encoding="utf-8",
                )
            )

        self.assertTrue(evaluate_mock.called)
        np.testing.assert_allclose(evaluate_mock.call_args.args[0], [1.0])
        self.assertTrue(evaluate_mock.call_args.kwargs["include_diagnostics"])
        self.assertAlmostEqual(payload["SEARCH_OBJECTIVE_J"], compact_eval["total"])
        self.assertAlmostEqual(payload["NONQS_RATIO"], rich_eval["J_QS"])
        self.assertAlmostEqual(payload["BOOZER_RESIDUAL"], rich_eval["J_Boozer"])
        self.assertAlmostEqual(payload["TOROIDAL_FLUX"], 0.24)
        self.assertAlmostEqual(payload["banana_surf_radius"], 0.142)
        self.assertAlmostEqual(payload["CURVATURE_THRESHOLD"], 100.0)
        self.assertEqual(payload["order"], 2)
        self.assertAlmostEqual(
            payload["BANANA_WINDING_SURFACE_MAJOR_RADIUS_M"],
            module.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
        )
        self.assertAlmostEqual(
            payload["COIL_WINDING_SURFACE_MAJOR_RADIUS_M"],
            module.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
        )

    def test_build_topology_gate_diagnostics_distinguishes_pass_reject_and_broken(self):
        module = load_single_stage_example_module()

        passed = module.build_topology_gate_diagnostics(
            {
                "enabled": True,
                "evaluated": True,
                "success": True,
                "state": "feasible",
                "survived_lines": 4,
                "nfieldlines": 4,
                "survival_fraction": 1.0,
                "survival_threshold": 0.5,
            },
            artifact_role="final_topology_gate",
        )
        rejected = module.build_topology_gate_diagnostics(
            {
                "enabled": True,
                "evaluated": True,
                "success": False,
                "state": "modeled_infeasible",
                "survived_lines": 1,
                "nfieldlines": 4,
                "survival_fraction": 0.25,
                "survival_threshold": 0.5,
                "first_exit_reason": "surface_exit",
                "first_exit_time": 0.2,
                "first_exit_angle": 0.1,
            },
            artifact_role="final_topology_gate",
        )
        broken = module.build_topology_gate_diagnostics(
            {
                "enabled": True,
                "evaluated": True,
                "success": False,
                "state": "broken",
                "broken": True,
                "evaluation_error": "trace exploded",
                "evaluation_error_type": "RuntimeError",
            },
            artifact_role="final_topology_gate",
        )

        self.assertEqual(passed["outcome"], "pass")
        self.assertEqual(passed["reason"], "survival_threshold_met")
        self.assertIn("Topology gate pass", passed["summary"])
        self.assertEqual(rejected["outcome"], "reject")
        self.assertEqual(rejected["reason"], "surface_exit")
        self.assertIn("surface_exit", rejected["summary"])
        self.assertEqual(broken["outcome"], "broken")
        self.assertEqual(broken["reason"], "RuntimeError")
        self.assertIn("trace exploded", broken["summary"])

    def test_build_topology_gate_diagnostics_keeps_skipped_gate_disabled(self):
        module = load_single_stage_example_module()

        diagnostics = module.build_topology_gate_diagnostics(
            module.skipped_topology_gate_status(),
            artifact_role="final_topology_gate",
        )

        self.assertEqual(diagnostics["outcome"], "not_evaluated")
        self.assertFalse(diagnostics["enabled"])
        self.assertFalse(diagnostics["evaluated"])
        self.assertEqual(diagnostics["reason"], "not_evaluated")

    def test_write_topology_checkpoint_artifacts_writes_diagnostics(self):
        module = load_single_stage_example_module()

        class FakeBiotSavart:
            def __init__(self):
                self.saved = []

            def save(self, path):
                self.saved.append(path)

        class FakeBoozerSurface:
            def __init__(self):
                self.surface = object()

        fake_bs = FakeBiotSavart()
        fake_outer = FakeBoozerSurface()
        topology_entry = {
            "accepted_iteration": 7,
            "topology_state": "evaluated",
            "topology_broken": False,
            "survived_lines": 10,
            "nfieldlines": 12,
            "confinement_score": 0.91,
            "confinement_loss": 0.08,
            "transport_diagnostics": {
                "schema_version": "single_stage_topology_transport_diagnostics_v1",
                "status": "partial",
                "effective_ripple": {
                    "status": "unavailable",
                    "aliases": ["epsilon_eff"],
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "best_topology"
            with patch.object(
                module, "save_surface_artifacts"
            ) as save_surface_artifacts:
                module.write_topology_checkpoint_artifacts(
                    out_dir,
                    artifact_role="best_topology_checkpoint",
                    topology_entry=topology_entry,
                    biotsavart=fake_bs,
                    surface_data=[{"name": "outer", "boozer_surface": fake_outer}],
                )

            diagnostics = json.loads(
                (out_dir / "topology_diagnostics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(diagnostics["kind"], "score")
            self.assertEqual(diagnostics["artifact_role"], "best_topology_checkpoint")
            self.assertEqual(diagnostics["outcome"], "scored")
            self.assertEqual(diagnostics["entry"]["accepted_iteration"], 7)
            self.assertEqual(
                diagnostics["entry"]["transport_diagnostics"]["effective_ripple"][
                    "aliases"
                ],
                ["epsilon_eff"],
            )
            self.assertEqual(fake_bs.saved, [str(out_dir / "biot_savart.json")])
            save_surface_artifacts.assert_called_once_with(
                [{"name": "outer", "boozer_surface": fake_outer}],
                fake_bs,
                out_dir,
                "surf",
                also_write_outer_legacy=False,
                boozer_state_interchange_manifest=None,
            )

    def test_build_preserved_timeout_results_payload_includes_replay_metadata(self):
        module = load_single_stage_example_module()
        replay_config = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_test.nc",
            plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
            stage2_seed_surf_path="/seeds/surf_opt_boozer_surface.json",
            strict_vacuum_current=True,
            strict_vacuum_seed_lineage="recent_stage1_candidate",
            stage1_candidate_id="s01_3240f0",
            strict_vacuum_source_current_group_projection="tf_banana_only",
            strict_vacuum_command_validation={"passed": True},
            strict_vacuum_seed_input_validation={
                "passed": True,
                "loaded_num_proxy_coils_zero_or_projected": True,
                "loaded_num_vf_coils_zero_or_projected": True,
            },
            residue_objective_replay_config=(
                ("enabled", True),
                ("weight", 0.25),
                ("targets_sha256", "sha256:targets"),
                ("seeds_sha256", "sha256:seeds"),
                ("target_manifest_id", "sha256:test-targets"),
                ("validation_id", "validation-artifact"),
                ("local_difference_step", 1.0e-6),
            ),
            winding_surface_free_mpol=1,
            winding_surface_free_ntor=1,
            winding_surface_free_dof_names=("rc(0,1)", "zs(0,1)"),
            coil_winding_surface_mpol=1,
            coil_winding_surface_ntor=1,
            coil_winding_surface_major_radius_m=0.920,
            banana_cws_embedded_winding_minor_radius_m=0.142,
            banana_cws_reembedded_on_live_surface=True,
            stage2_policy_metadata=module.stage2_policy_metadata_items(
                {
                    "BOOZER_CURRENT_CONVENTION": "mu0",
                    "BOOZER_I": -0.008168140899333462,
                    "G0_POLICY": "signed_explicit_tf_current",
                    "PROXY_PLACEMENT_MODE": "surface_major_radius_z0",
                    "PROXY_VF_CURRENT_SCALAR_POLICY": "signed_physical_scalar",
                    "PROXY_PLASMA_CURRENT_A": -6.5e3,
                    "VF_CURRENT_A": -1.0e3,
                    "VF_CURRENT_MAX_A": 1.6e4,
                    "VF_TEMPLATE_PATH": "/seeds/vf.json",
                    "VF_TEMPLATE_SHA256": "abc123",
                    "VF_CURRENT_SIGN_POLICY": "template_sign_abs_proxy_current",
                    "VF_CURRENT_MUTABILITY": "shared_unfixed_scaled_current",
                    "FLIP_BANANA": True,
                    "BANANA_CURRENT_SIGN": -1,
                    "BANANA_CURRENT_PINNED": True,
                    "BANANA_I_FIXED_S2_KA": -14.0,
                    "IOTA_TARGET_SIGN": -1,
                }
            ),
        )
        run_dict = {
            "search_eval": {
                "total": 7.5e-4,
                "base_total": 7.4e-4,
            },
            "J": 7.5e-4,
            "intersecting": False,
            "surface_status": {"success": True},
            "accepted_hardware_status": {"success": False},
            "topology_gate_status": {"success": True},
            "best_topology": {
                "accepted_iteration": 1,
                "confinement_score": 0.91,
                "confinement_loss": 0.08,
                "invariant_torus_fraction": 0.5,
                "kam_fraction": 0.5,
                "kam_fraction_semantics": KAM_FRACTION_SEMANTICS,
                "kam_median_width": 0.08,
                "cross_section_span": 0.2,
                "frontier_certification_enabled": True,
                "frontier_certification_ok": True,
                "frontier_certification_reason": "certified",
                "frontier_certification_hardware_ok": True,
                "frontier_invariant_torus_min": 0.30,
                "frontier_invariant_torus_deficit": 0.0,
                "frontier_kam_min": 0.30,
                "frontier_kam_deficit": 0.0,
            },
        }
        payload = module.build_preserved_timeout_results_payload(
            replay_config=replay_config,
            preservation_kind="best_accepted",
            incumbent_stage="initial",
            run_dict=run_dict,
            objective_eval={"J_QS": 2.7e-4, "J_Boozer": 4.8e-7},
            field_error=3.5e-4,
            final_iota=0.14997,
            final_volume=0.09998,
            hardware_snapshot={
                "search_hardware_status": {
                    "success": False,
                    "violations": [
                        "coil_coil_spacing 0.049600 below threshold 0.050000"
                    ],
                },
                "artifact_hardware_status": {
                    "success": False,
                    "violations": [
                        "coil_coil_spacing 0.049600 below threshold 0.050000"
                    ],
                },
                "max_curvature": 19.8,
                "length_target": 1.7,
                "tf_current_A": -8.0e4,
                "tf_current_limit_A": 8.0e4,
                "banana_current_A": 1.4e4,
                "banana_current_max_A": 1.6e4,
                "curve_curve_min_dist": 0.0496,
                "curve_surface_min_dist": 0.067,
                "surface_vessel_min_dist": 0.082,
            },
            coil_length=2.91,
            accepted_iteration=1,
        )

        self.assertEqual(payload["PLASMA_SURF_PATH"], str(SIGNED_CW_WOUT_PATH))
        self.assertEqual(payload["SEED_ARTIFACT_ROLE"], "stage2")
        self.assertFalse(payload["OFFSPEC_REPLAY_DEBUG_ONLY"])
        self.assertIsNone(payload["SINGLE_STAGE_RESUME_BS_PATH"])
        self.assertEqual(payload["CURRENT_LINEAGE"], "strict_vacuum")
        self.assertTrue(payload["STRICT_VACUUM_CURRENT"])
        self.assertEqual(
            payload["STRICT_VACUUM_SEED_LINEAGE"], "recent_stage1_candidate"
        )
        self.assertEqual(payload["STAGE1_CANDIDATE_ID"], "s01_3240f0")
        self.assertTrue(payload["STRICT_VACUUM_PRODUCTION_CANDIDATE"])
        self.assertFalse(payload["STRICT_VACUUM_CONTROL_ONLY"])
        self.assertEqual(
            payload["STRICT_VACUUM_SOURCE_CURRENT_GROUP_PROJECTION"],
            "tf_banana_only",
        )
        self.assertEqual(payload["STRICT_VACUUM_COMMAND_VALIDATION"], {"passed": True})
        self.assertEqual(
            payload["STRICT_VACUUM_SEED_INPUT_VALIDATION"],
            {
                "passed": True,
                "loaded_num_proxy_coils_zero_or_projected": True,
                "loaded_num_vf_coils_zero_or_projected": True,
            },
        )
        self.assertEqual(payload["EFFECTIVE_CURRENT_MODE"], "vacuum")
        self.assertIsNone(payload["FINITE_CURRENT_MODE"])
        self.assertNotIn(module.DESIGN_ONLY_RESULTS_KEY, payload)
        self.assertEqual(payload["NUM_PROXY_COILS"], 0)
        self.assertEqual(payload["NUM_VF_COILS"], 0)
        self.assertEqual(payload["PROXY_PLASMA_CURRENT_A"], 0.0)
        self.assertEqual(payload["VF_CURRENT_A"], 0.0)
        self.assertEqual(payload["STAGE2_BS_PATH"], "/seeds/biot_savart_opt.json")
        self.assertEqual(payload["STAGE2_BOOZER_CURRENT_CONVENTION"], "mu0")
        self.assertEqual(payload["STAGE2_BOOZER_I"], -0.008168140899333462)
        self.assertEqual(payload["STAGE2_G0_POLICY"], "signed_explicit_tf_current")
        self.assertEqual(
            payload["STAGE2_PROXY_PLACEMENT_MODE"], "surface_major_radius_z0"
        )
        self.assertEqual(
            payload["STAGE2_PROXY_VF_CURRENT_SCALAR_POLICY"],
            "signed_physical_scalar",
        )
        self.assertEqual(payload["STAGE2_PROXY_PLASMA_CURRENT_A"], -6.5e3)
        self.assertEqual(payload["STAGE2_VF_CURRENT_A"], -1.0e3)
        self.assertEqual(payload["STAGE2_VF_CURRENT_MAX_A"], 1.6e4)
        self.assertEqual(payload["STAGE2_VF_TEMPLATE_PATH"], "/seeds/vf.json")
        self.assertEqual(payload["STAGE2_VF_TEMPLATE_SHA256"], "abc123")
        self.assertEqual(
            payload["STAGE2_VF_CURRENT_SIGN_POLICY"],
            "template_sign_abs_proxy_current",
        )
        self.assertEqual(
            payload["STAGE2_VF_CURRENT_MUTABILITY"],
            "shared_unfixed_scaled_current",
        )
        self.assertTrue(payload["STAGE2_FLIP_BANANA"])
        self.assertEqual(payload["STAGE2_BANANA_CURRENT_SIGN"], -1)
        self.assertTrue(payload["STAGE2_BANANA_CURRENT_PINNED"])
        self.assertEqual(payload["STAGE2_BANANA_I_FIXED_S2_KA"], -14.0)
        self.assertEqual(payload["STAGE2_IOTA_TARGET_SIGN"], -1)
        self.assertEqual(payload["WINDING_SURFACE_FREE_MPOL"], 1)
        self.assertEqual(payload["WINDING_SURFACE_FREE_NTOR"], 1)
        self.assertEqual(
            payload["WINDING_SURFACE_FREE_DOF_NAMES"],
            ["rc(0,1)", "zs(0,1)"],
        )
        self.assertEqual(payload["COIL_WINDING_SURFACE_MPOL"], 1)
        self.assertEqual(payload["COIL_WINDING_SURFACE_NTOR"], 1)
        self.assertEqual(payload["BANANA_WINDING_SURFACE_MAJOR_RADIUS_M"], 0.920)
        self.assertEqual(payload["COIL_WINDING_SURFACE_MAJOR_RADIUS_M"], 0.920)
        self.assertEqual(payload["BANANA_CWS_EMBEDDED_WINDING_MINOR_RADIUS_M"], 0.142)
        self.assertTrue(payload["BANANA_CWS_REEMBEDDED_ON_LIVE_SURFACE"])
        self.assertEqual(
            payload["GREENE_RESIDUE_OBJECTIVE_REPLAY_CONFIG"],
            {
                "enabled": True,
                "weight": 0.25,
                "targets_sha256": "sha256:targets",
                "seeds_sha256": "sha256:seeds",
                "target_manifest_id": "sha256:test-targets",
                "validation_id": "validation-artifact",
                "local_difference_step": 1.0e-6,
            },
        )
        self.assertEqual(
            payload["STAGE2_SEED_SURF_PATH"],
            "/seeds/surf_opt_boozer_surface.json",
        )
        self.assertEqual(payload["STAGE2_RESULTS_PATH"], "/seeds/results.json")
        self.assertEqual(payload["mpol"], 8)
        self.assertEqual(payload["ntor"], 6)
        self.assertEqual(payload["nphi"], 127)
        self.assertEqual(payload["ntheta"], 32)
        self.assertEqual(payload["CONSTRAINT_WEIGHT"], 1.0)
        self.assertEqual(payload["max_iterations"], 30)
        self.assertEqual(payload["TARGET_VOLUME"], 0.10)
        self.assertEqual(payload["TARGET_IOTA"], 0.15)
        self.assertEqual(
            payload["FINAL_SOURCE_STAGE"], payload["PRESERVED_TIMEOUT_SALVAGE_STAGE"]
        )
        self.assertEqual(payload["FINAL_TOPOLOGY_GATE_DIAGNOSTICS"]["kind"], "gate")
        self.assertEqual(
            payload["FINAL_TOPOLOGY_GATE_DIAGNOSTICS"]["outcome"],
            "pass",
        )
        self.assertEqual(payload["MAX_CURVATURE"], 19.8)
        self.assertEqual(payload["COIL_LENGTH"], 2.91)
        self.assertEqual(payload["LENGTH_TARGET"], 1.7)
        self.assertEqual(payload["TF_CURRENT_A"], -8.0e4)
        self.assertEqual(payload["TF_CURRENT_LIMIT_A"], 8.0e4)
        self.assertEqual(payload["BANANA_CURRENT_A"], 1.4e4)
        self.assertEqual(payload["BANANA_CURRENT_MAX_A"], 1.6e4)
        self.assertIsNone(payload["FINAL_TOPOLOGY_TRANSPORT_DIAGNOSTICS"])
        self.assertEqual(payload["BEST_TOPOLOGY_ACCEPTED_ITERATION"], 1)
        self.assertAlmostEqual(payload["BEST_TOPOLOGY_KAM_FRACTION"], 0.5)
        self.assertEqual(
            payload["BEST_TOPOLOGY_KAM_FRACTION_SEMANTICS"],
            KAM_FRACTION_SEMANTICS,
        )
        self.assertTrue(payload["BEST_TOPOLOGY_CERTIFICATION_OK"])

    def test_inherited_proxy_metadata_persists_design_only_marker(self):
        module = load_single_stage_example_module()
        reason = "finite_current_proxy_line_current: wataru_proxy_field"
        stage2_metadata = module.stage2_policy_metadata_items(
            {
                **module.build_design_only_results_fields(reason=reason),
                "FINITE_CURRENT_MODE": "wataru_proxy_field",
            }
        )

        current_payload = module.inherited_design_only_results_payload(
            stage2_metadata,
            current_num_proxy_coils=1,
            finite_current_mode="wataru_proxy_field",
        )
        strict_vacuum_payload = module.inherited_design_only_results_payload(
            stage2_metadata,
            current_num_proxy_coils=0,
            finite_current_mode="wataru_proxy_field",
        )
        provenance_payload = module.stage2_policy_metadata_payload(stage2_metadata)

        self.assertIs(current_payload[module.DESIGN_ONLY_RESULTS_KEY], True)
        self.assertEqual(current_payload[module.DESIGN_ONLY_REASON_KEY], reason)
        self.assertEqual(strict_vacuum_payload, {})
        self.assertIs(
            provenance_payload[f"STAGE2_{module.DESIGN_ONLY_RESULTS_KEY}"],
            True,
        )

    def test_mark_inherited_proxy_biot_savart_fails_closed_in_process_score(self):
        module = load_single_stage_example_module()
        reason = "finite_current_proxy_line_current: wataru_proxy_field"
        stage2_metadata = module.stage2_policy_metadata_items(
            {
                **module.build_design_only_results_fields(reason=reason),
                "FINITE_CURRENT_MODE": "wataru_proxy_field",
            }
        )
        bs = SimpleNamespace()

        payload = module.mark_inherited_design_only_biot_savart(
            bs,
            stage2_metadata,
            current_num_proxy_coils=1,
            finite_current_mode="wataru_proxy_field",
        )
        result = module.safe_score_topology(
            object(),
            bs,
            nfieldlines=1,
            tmax=1.0,
        )

        self.assertEqual(payload[module.DESIGN_ONLY_REASON_KEY], reason)
        self.assertTrue(getattr(bs, "_design_only_no_topology_gate"))
        self.assertTrue(result["broken"])
        self.assertEqual(
            result["evaluation_error_type"],
            "DesignOnlyTopologyFieldError",
        )

    def test_preserved_timeout_proxy_payload_persists_design_only_marker(self):
        module = load_single_stage_example_module()
        reason = "finite_current_proxy_line_current: wataru_proxy_field"
        replay_config = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_10x10.nc",
            plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
            stage2_seed_surf_path="/seeds/surf_opt_boozer_surface.json",
            strict_vacuum_current=False,
            finite_current_mode="wataru_proxy_field",
            effective_current_mode="wataru_proxy_field",
            boozer_current_convention="mu0",
            plasma_current_A=800.0,
            boozer_I=1.0053096491487339e-3,
            num_proxy_coils=1,
            num_vf_coils=0,
            proxy_plasma_current_A=800.0,
            vf_current_A=0.0,
            stage2_policy_metadata=module.stage2_policy_metadata_items(
                {
                    **module.build_design_only_results_fields(reason=reason),
                    "BOOZER_CURRENT_CONVENTION": "mu0",
                    "FINITE_CURRENT_MODE": "wataru_proxy_field",
                }
            ),
        )
        run_dict = {
            "search_eval": {
                "total": 7.5e-4,
                "base_total": 7.4e-4,
            },
            "J": 7.5e-4,
            "intersecting": False,
            "surface_status": {"success": True},
            "accepted_hardware_status": {"success": True},
            "topology_gate_status": {"success": False, "state": "broken"},
        }
        payload = module.build_preserved_timeout_results_payload(
            replay_config=replay_config,
            preservation_kind="best_accepted",
            incumbent_stage="initial",
            run_dict=run_dict,
            objective_eval={"J_QS": 2.7e-4, "J_Boozer": 4.8e-7},
            field_error=3.5e-4,
            final_iota=0.14997,
            final_volume=0.09998,
            hardware_snapshot={
                "search_hardware_status": {"success": True, "violations": []},
                "artifact_hardware_status": {"success": True, "violations": []},
                "tf_current_A": -8.0e4,
            },
            coil_length=1.91,
            accepted_iteration=1,
        )

        self.assertIs(payload[module.DESIGN_ONLY_RESULTS_KEY], True)
        self.assertEqual(payload[module.DESIGN_ONLY_REASON_KEY], reason)
        self.assertIs(payload[f"STAGE2_{module.DESIGN_ONLY_RESULTS_KEY}"], True)
        self.assertEqual(payload["NUM_PROXY_COILS"], 1)

    def test_build_preserved_timeout_results_payload_stamps_producer_wout_convention(
        self,
    ):
        # Producer-side stamping of WOUT_CONVENTION / WOUT_OFF_SPEC for the
        # preserved-timeout sidecar — proven via build_preserved_timeout_results_payload
        # alone, without the consumer-side legacy upgrader being invoked. Mirrors
        # the Stage 2 producer stamp emitted by ``banana_coil_solver``.
        module = load_single_stage_example_module()
        replay_config = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_10x10.nc",
            plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
            stage2_seed_surf_path="/seeds/surf_opt_boozer_surface.json",
        )
        run_dict = {
            "search_eval": {
                "total": 7.5e-4,
                "base_total": 7.4e-4,
            },
            "J": 7.5e-4,
            "intersecting": False,
            "surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"success": True},
        }
        payload = module.build_preserved_timeout_results_payload(
            replay_config=replay_config,
            preservation_kind="best_feasible",
            incumbent_stage="initial",
            run_dict=run_dict,
            objective_eval={"J_QS": 2.7e-4, "J_Boozer": 4.8e-7},
            field_error=3.5e-4,
            final_iota=0.14997,
            final_volume=0.09998,
            hardware_snapshot={
                "search_hardware_status": {"success": True, "violations": []},
                "artifact_hardware_status": {"success": True, "violations": []},
                "max_curvature": 19.8,
                "tf_current_A": -8.0e4,
                "curve_curve_min_dist": 0.0496,
                "curve_surface_min_dist": 0.067,
                "surface_vessel_min_dist": 0.082,
            },
            coil_length=2.91,
            accepted_iteration=1,
        )

        # Stamp matches the WOUT lane (signed_cw fixture) and the negative
        # TF-current sign: WOUT_OFF_SPEC must be False.
        self.assertEqual(payload["WOUT_CONVENTION"], "signed_cw")
        self.assertIs(payload["WOUT_OFF_SPEC"], False)

    def test_build_preserved_timeout_results_payload_stamps_offspec_when_tf_lane_disagrees(
        self,
    ):
        # Same WOUT fixture but a positive TF current: the producer must stamp
        # WOUT_OFF_SPEC=True at write time rather than masking the mismatch.
        module = load_single_stage_example_module()
        replay_config = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_10x10.nc",
            plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
        )
        run_dict = {
            "search_eval": {
                "total": 7.5e-4,
                "base_total": 7.4e-4,
            },
            "J": 7.5e-4,
            "intersecting": False,
            "surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"success": True},
        }
        payload = module.build_preserved_timeout_results_payload(
            replay_config=replay_config,
            preservation_kind="best_feasible",
            incumbent_stage="initial",
            run_dict=run_dict,
            objective_eval={"J_QS": 2.7e-4, "J_Boozer": 4.8e-7},
            field_error=3.5e-4,
            final_iota=0.14997,
            final_volume=0.09998,
            hardware_snapshot={
                "search_hardware_status": {"success": True, "violations": []},
                "artifact_hardware_status": {"success": True, "violations": []},
                "max_curvature": 19.8,
                "tf_current_A": 8.0e4,
                "curve_curve_min_dist": 0.0496,
                "curve_surface_min_dist": 0.067,
                "surface_vessel_min_dist": 0.082,
            },
            coil_length=2.91,
            accepted_iteration=1,
        )

        self.assertEqual(payload["WOUT_CONVENTION"], "signed_cw")
        self.assertIs(payload["WOUT_OFF_SPEC"], True)

    def test_single_stage_final_results_write_stamps_producer_wout_convention(self):
        # Producer-side wiring check for the final results.json write at end of
        # main(): the file's source must call ``wout_convention_artifact_fields``
        # immediately before ``write_json_artifact(..., "results.json", ...)`` so
        # the stamp lands at producer time, independent of the consumer-side
        # legacy upgrader. This test guards against the wiring being removed.
        source = EXAMPLE_MODULE_PATH.read_text(encoding="utf-8")
        # The final results.json write is hoisted: ``results_path =
        # os.path.join(OUT_DIR_ITER, "results.json")`` then
        # ``write_json_artifact(results_path, results)`` (unique in the module).
        write_idx = source.index("write_json_artifact(results_path, results)")
        # Window the lookback so unrelated call sites elsewhere in the file
        # cannot spoof a passing assertion, with margin above the current
        # stamp->write distance (~930 chars) for intervening result fields.
        window = source[max(0, write_idx - 1500) : write_idx]
        self.assertIn("wout_convention_artifact_fields(", window)
        self.assertIn("wout_path=file_loc", window)
        self.assertIn("tf_current_A=stage2_tf_current_A", window)

    def test_build_preserved_timeout_results_payload_reports_banana_current_mode_fields(
        self,
    ):
        module = load_single_stage_example_module()
        replay_config = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_test.nc",
            plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
            single_stage_banana_current_mode="independent",
            num_banana_current_controls=2,
        )
        run_dict = {
            "search_eval": {
                "total": 7.5e-4,
                "base_total": 7.4e-4,
            },
            "J": 7.5e-4,
            "intersecting": False,
            "surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"success": True},
        }
        banana_current_state = module.SingleStageBananaCurrentState(
            mode="independent",
            currents=(Current(1.2e4), Current(-1.5e4)),
            seed_currents_A=(1.0e4, -1.0e4),
        )

        payload = module.build_preserved_timeout_results_payload(
            replay_config=replay_config,
            preservation_kind="best_feasible",
            incumbent_stage="initial",
            run_dict=run_dict,
            objective_eval={"J_QS": 2.7e-4, "J_Boozer": 4.8e-7},
            field_error=3.5e-4,
            final_iota=0.14997,
            final_volume=0.09998,
            hardware_snapshot={
                "search_hardware_status": {"success": True, "violations": []},
                "artifact_hardware_status": {"success": True, "violations": []},
                "max_curvature": 19.8,
                "tf_current_A": -8.0e4,
                "curve_curve_min_dist": 0.0496,
                "curve_surface_min_dist": 0.067,
                "surface_vessel_min_dist": 0.082,
                "banana_current_A": 1.5e4,
                "banana_current_max_A": 1.6e4,
            },
            banana_current_state=banana_current_state,
            coil_length=2.91,
            accepted_iteration=1,
        )

        self.assertEqual(payload["BANANA_CURRENT_MODE"], "independent")
        self.assertEqual(payload["BANANA_CURRENTS_A"], [1.2e4, -1.5e4])
        self.assertEqual(payload["BANANA_CURRENT_MAX_ABS_A"], 1.5e4)
        self.assertEqual(payload["BANANA_CURRENT_CONTROL_METRIC"], "max_abs")
        self.assertEqual(payload["BANANA_NUM_CURRENT_CONTROLS"], 2)
        self.assertEqual(payload["BANANA_CURRENT_A"], 1.5e4)

    def test_build_preserved_timeout_results_payload_includes_alm_runtime_state(self):
        module = load_single_stage_example_module()
        replay_config = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_test.nc",
            plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="alm",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
        )
        run_dict = {
            "search_eval": {
                "total": 9.5e-4,
                "base_total": 8.1e-4,
                "max_feasibility_violation": 4.0e-3,
                "metric_stationarity_norm": 7.5e-5,
                "constraint_names": [
                    "coil_length_upper_bound",
                    "banana_current_upper_bound",
                ],
                "constraint_scales": np.array([2.0, 16000.0]),
                "constraint_blocks": ["geometry", "current"],
                "constraint_scale_sources": [
                    "threshold:coil_length_upper_bound",
                    "threshold:banana_current_upper_bound",
                ],
                "constraint_values": np.array([1.0e-3, -2.0e-4]),
                "raw_feasibility_values": np.array([1.6e1, 0.0]),
                "normalized_feasibility_values": np.array([1.0e-3, 0.0]),
                "raw_dual_update_values": np.array([9.6, -1.6]),
                "dual_update_values": np.array([7.0e-4, -1.1e-4]),
                "raw_hard_dual_update_values": np.array([13.5, -2.2]),
                "hard_dual_update_values": np.array([8.5e-4, -1.4e-4]),
                "raw_hard_signed_constraint_values": np.array([14.0, -2.5]),
                "hard_signed_constraint_values": np.array([8.75e-4, -1.5625e-4]),
                "raw_hard_violation_values": np.array([14.0, 0.0]),
                "hard_violation_values": np.array([8.75e-4, 0.0]),
                "raw_surrogate_signed_constraint_values": np.array([16.0, -3.2]),
                "surrogate_signed_constraint_values": np.array([1.0e-3, -2.0e-4]),
                "hard_max_violation": 8.75e-4,
                "surrogate_max_value": 1.0e-3,
                "hard_positive_shift_zero": False,
                "signal_mismatch_active": True,
                "penalty_gradient_norm": 3.3e-2,
                "trust_radius": 0.25,
            },
            "J": 9.5e-4,
            "intersecting": False,
            "surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"success": True},
            "alm_outer_iteration": 3,
            "alm_feasibility_tolerance": 1.0e-4,
            "alm_stationarity_tolerance": 2.0e-4,
        }
        with patch.object(
            module,
            "build_alm_final_constraint_payload",
            wraps=module.build_alm_final_constraint_payload,
        ) as build_constraint_payload:
            payload = module.build_preserved_timeout_results_payload(
                replay_config=replay_config,
                preservation_kind="best_feasible",
                incumbent_stage="final",
                run_dict=run_dict,
                objective_eval={"J_QS": 1.7e-4, "J_Boozer": 8.0e-7},
                field_error=2.5e-4,
                final_iota=0.151,
                final_volume=0.101,
                hardware_snapshot={
                    "search_hardware_status": {"success": True, "violations": []},
                    "artifact_hardware_status": {"success": True, "violations": []},
                    "max_curvature": 18.2,
                    "tf_current_A": -8.0e4,
                    "curve_curve_min_dist": 0.051,
                    "curve_surface_min_dist": 0.068,
                    "surface_vessel_min_dist": 0.084,
                },
                coil_length=2.85,
                accepted_iteration=4,
                alm_runtime_state=module.build_preserved_timeout_alm_state(
                    constraint_method="alm",
                    penalty=12.5,
                    multipliers=np.array([0.25, -0.75]),
                ),
            )

        self.assertEqual(payload["ALM_FORMULATION"], "weighted_sum")
        self.assertEqual(build_constraint_payload.call_count, 1)
        self.assertEqual(payload["ALM_OUTER_ITERATIONS"], 3)
        self.assertEqual(payload["ALM_FINAL_PENALTY"], 12.5)
        self.assertEqual(payload["ALM_FINAL_MULTIPLIERS"], [0.25, -0.75])
        np.testing.assert_allclose(
            payload["ALM_FINAL_RAW_DUAL_ESTIMATES"],
            [0.125, -4.6875e-5],
        )
        self.assertEqual(
            payload["ALM_CONSTRAINT_NAMES"],
            ["coil_length_upper_bound", "banana_current_upper_bound"],
        )
        self.assertEqual(payload["ALM_CONSTRAINT_SCALES"], [2.0, 16000.0])
        self.assertEqual(payload["ALM_CONSTRAINT_BLOCKS"], ["geometry", "current"])
        self.assertEqual(
            payload["ALM_CONSTRAINT_SCALE_SOURCES"],
            [
                "threshold:coil_length_upper_bound",
                "threshold:banana_current_upper_bound",
            ],
        )
        self.assertEqual(payload["ALM_FINAL_CONSTRAINT_VALUES"], [16.0, -3.2])
        self.assertEqual(
            payload["ALM_FINAL_NORMALIZED_CONSTRAINT_VALUES"],
            [1.0e-3, -2.0e-4],
        )
        self.assertEqual(payload["ALM_FINAL_SOLVER_CONSTRAINT_VALUES"], [16.0, -3.2])
        self.assertEqual(
            payload["ALM_FINAL_NORMALIZED_SOLVER_CONSTRAINT_VALUES"],
            [1.0e-3, -2.0e-4],
        )
        self.assertEqual(
            payload["ALM_FINAL_HARD_SIGNED_CONSTRAINT_VALUES"], [14.0, -2.5]
        )
        self.assertEqual(
            payload["ALM_FINAL_NORMALIZED_HARD_SIGNED_CONSTRAINT_VALUES"],
            [8.75e-4, -1.5625e-4],
        )
        self.assertEqual(payload["ALM_FINAL_HARD_VIOLATION_VALUES"], [14.0, 0.0])
        self.assertEqual(
            payload["ALM_FINAL_NORMALIZED_HARD_VIOLATION_VALUES"],
            [8.75e-4, 0.0],
        )
        self.assertEqual(
            payload["ALM_FINAL_SURROGATE_SIGNED_CONSTRAINT_VALUES"],
            [16.0, -3.2],
        )
        self.assertEqual(
            payload["ALM_FINAL_NORMALIZED_SURROGATE_SIGNED_CONSTRAINT_VALUES"],
            [1.0e-3, -2.0e-4],
        )
        self.assertEqual(payload["ALM_FINAL_HARD_MAX_VIOLATION"], 8.75e-4)
        self.assertEqual(payload["ALM_FINAL_SURROGATE_MAX_VALUE"], 1.0e-3)
        self.assertFalse(payload["ALM_FINAL_HARD_POSITIVE_SHIFT_ZERO"])
        self.assertTrue(payload["ALM_FINAL_SIGNAL_MISMATCH_ACTIVE"])
        self.assertEqual(payload["ALM_FINAL_PENALTY_GRADIENT_NORM"], 3.3e-2)
        self.assertEqual(payload["ALM_FINAL_TRUST_RADIUS"], 0.25)
        self.assertEqual(payload["ALM_FINAL_FEASIBILITY_TOL"], 1.0e-4)
        self.assertEqual(payload["ALM_FINAL_STATIONARITY_TOL"], 2.0e-4)
        self.assertEqual(payload["ALM_FINAL_MAX_FEASIBILITY_VIOLATION"], 4.0e-3)
        self.assertEqual(payload["ALM_FINAL_STATIONARITY_NORM"], 7.5e-5)

    def test_build_alm_final_constraint_payload_emits_hard_surrogate_sidecars(self):
        module = load_single_stage_example_module()
        payload = module.build_alm_final_constraint_payload(
            SimpleNamespace(
                constraint_names=["coil_surface_spacing"],
                raw_dual_estimates=[0.002],
                constraint_scales=[0.02],
                constraint_blocks=["geometry"],
                constraint_scale_sources=["threshold:coil_surface_spacing"],
                raw_constraint_values=[0.01],
                normalized_constraint_values=[0.5],
                raw_solver_constraint_values=[0.2],
                normalized_solver_constraint_values=[10.0],
                raw_hard_signed_constraint_values=[0.01],
                hard_signed_constraint_values=[0.5],
                raw_hard_violation_values=[0.01],
                hard_violation_values=[0.5],
                raw_surrogate_signed_constraint_values=[0.2],
                surrogate_signed_constraint_values=[10.0],
                final_hard_max_violation=0.01,
                final_surrogate_max_value=10.0,
                hard_positive_shift_zero=False,
                signal_mismatch_active=True,
                final_penalty_gradient_norm=0.4,
                trust_radius=0.25,
            )
        )

        self.assertEqual(payload["ALM_CONSTRAINT_NAMES"], ["coil_surface_spacing"])
        self.assertEqual(payload["ALM_FINAL_RAW_DUAL_ESTIMATES"], [0.002])
        self.assertEqual(payload["ALM_CONSTRAINT_SCALES"], [0.02])
        self.assertEqual(payload["ALM_CONSTRAINT_BLOCKS"], ["geometry"])
        self.assertEqual(
            payload["ALM_CONSTRAINT_SCALE_SOURCES"],
            ["threshold:coil_surface_spacing"],
        )
        self.assertEqual(payload["ALM_FINAL_CONSTRAINT_VALUES"], [0.01])
        self.assertEqual(payload["ALM_FINAL_NORMALIZED_CONSTRAINT_VALUES"], [0.5])
        self.assertEqual(payload["ALM_FINAL_SOLVER_CONSTRAINT_VALUES"], [0.2])
        self.assertEqual(
            payload["ALM_FINAL_NORMALIZED_SOLVER_CONSTRAINT_VALUES"],
            [10.0],
        )
        self.assertEqual(payload["ALM_FINAL_HARD_SIGNED_CONSTRAINT_VALUES"], [0.01])
        self.assertEqual(
            payload["ALM_FINAL_NORMALIZED_HARD_SIGNED_CONSTRAINT_VALUES"],
            [0.5],
        )
        self.assertEqual(payload["ALM_FINAL_HARD_VIOLATION_VALUES"], [0.01])
        self.assertEqual(
            payload["ALM_FINAL_RAW_HARD_VIOLATION_BY_CONSTRAINT"],
            [0.01],
        )
        self.assertEqual(payload["ALM_FINAL_NORMALIZED_HARD_VIOLATION_VALUES"], [0.5])
        self.assertEqual(
            payload["ALM_FINAL_SURROGATE_SIGNED_CONSTRAINT_VALUES"],
            [0.2],
        )
        self.assertEqual(
            payload["ALM_FINAL_NORMALIZED_SURROGATE_SIGNED_CONSTRAINT_VALUES"],
            [10.0],
        )
        self.assertEqual(payload["ALM_FINAL_HARD_MAX_VIOLATION"], 0.01)
        self.assertEqual(payload["ALM_FINAL_SURROGATE_MAX_VALUE"], 10.0)
        self.assertFalse(payload["ALM_FINAL_HARD_POSITIVE_SHIFT_ZERO"])
        self.assertTrue(payload["ALM_FINAL_SIGNAL_MISMATCH_ACTIVE"])
        self.assertEqual(payload["ALM_FINAL_PENALTY_GRADIENT_NORM"], 0.4)
        self.assertEqual(payload["ALM_FINAL_TRUST_RADIUS"], 0.25)

    def test_build_alm_final_constraint_payload_does_not_label_normalized_values_as_raw(
        self,
    ):
        module = load_single_stage_example_module()
        payload = module.build_alm_final_constraint_payload(
            SimpleNamespace(
                constraint_names=["coil_surface_spacing"],
                raw_dual_estimates=None,
                constraint_scales=[0.02],
                constraint_blocks=["geometry"],
                constraint_scale_sources=["threshold:coil_surface_spacing"],
                normalized_constraint_values=[0.5],
                normalized_solver_constraint_values=[10.0],
                hard_signed_constraint_values=[0.5],
                hard_violation_values=[0.5],
                surrogate_signed_constraint_values=[10.0],
                final_hard_max_violation=0.5,
                final_surrogate_max_value=10.0,
                hard_positive_shift_zero=False,
                signal_mismatch_active=True,
                final_penalty_gradient_norm=0.4,
                trust_radius=0.25,
            )
        )

        self.assertIsNone(payload["ALM_FINAL_CONSTRAINT_VALUES"])
        self.assertEqual(payload["ALM_FINAL_NORMALIZED_CONSTRAINT_VALUES"], [0.5])
        self.assertIsNone(payload["ALM_FINAL_SOLVER_CONSTRAINT_VALUES"])
        self.assertEqual(
            payload["ALM_FINAL_NORMALIZED_SOLVER_CONSTRAINT_VALUES"], [10.0]
        )
        self.assertIsNone(payload["ALM_FINAL_HARD_SIGNED_CONSTRAINT_VALUES"])
        self.assertEqual(
            payload["ALM_FINAL_NORMALIZED_HARD_SIGNED_CONSTRAINT_VALUES"], [0.5]
        )
        self.assertIsNone(payload["ALM_FINAL_HARD_VIOLATION_VALUES"])
        self.assertEqual(payload["ALM_FINAL_NORMALIZED_HARD_VIOLATION_VALUES"], [0.5])
        self.assertIsNone(payload["ALM_FINAL_SURROGATE_SIGNED_CONSTRAINT_VALUES"])
        self.assertEqual(
            payload["ALM_FINAL_NORMALIZED_SURROGATE_SIGNED_CONSTRAINT_VALUES"],
            [10.0],
        )

    def test_alm_result_view_from_search_eval_uses_only_explicit_raw_sidecars(self):
        module = load_single_stage_example_module()
        result_view = module._alm_result_view_from_search_eval(
            {
                "constraint_names": ["coil_surface_spacing"],
                "constraint_scales": np.array([0.02]),
                "constraint_values": np.array([10.0]),
                "normalized_signed_constraint_values": np.array([0.5]),
                "hard_signed_constraint_values": np.array([0.5]),
                "hard_violation_values": np.array([0.5]),
                "surrogate_signed_constraint_values": np.array([10.0]),
            },
            np.array([0.25]),
        )

        self.assertIsNone(result_view.raw_constraint_values)
        self.assertEqual(result_view.normalized_constraint_values.tolist(), [0.5])
        self.assertIsNone(result_view.raw_solver_constraint_values)
        self.assertEqual(
            result_view.normalized_solver_constraint_values.tolist(), [10.0]
        )
        self.assertIsNone(result_view.raw_hard_signed_constraint_values)
        self.assertEqual(result_view.hard_signed_constraint_values.tolist(), [0.5])
        self.assertIsNone(result_view.raw_hard_violation_values)
        self.assertEqual(result_view.hard_violation_values.tolist(), [0.5])
        self.assertIsNone(result_view.raw_surrogate_signed_constraint_values)
        self.assertEqual(
            result_view.surrogate_signed_constraint_values.tolist(), [10.0]
        )

    def test_build_preserved_timeout_results_payload_uses_artifact_hardware_status_for_final_feasibility(
        self,
    ):
        module = load_single_stage_example_module()
        replay_config = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_test.nc",
            plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
        )
        run_dict = {
            "search_eval": {
                "total": 7.5e-4,
                "base_total": 7.4e-4,
            },
            "J": 7.5e-4,
            "intersecting": False,
            "surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"success": True},
        }

        payload = module.build_preserved_timeout_results_payload(
            replay_config=replay_config,
            preservation_kind="best_accepted",
            incumbent_stage="initial",
            run_dict=run_dict,
            objective_eval={"J_QS": 2.7e-4, "J_Boozer": 4.8e-7},
            field_error=3.5e-4,
            final_iota=0.14997,
            final_volume=0.09998,
            hardware_snapshot={
                "search_hardware_status": {"success": True, "violations": []},
                "artifact_hardware_status": {
                    "success": False,
                    "violations": ["coil_length 2.100000 exceeds threshold 2.000000"],
                },
                "max_curvature": 19.8,
                "tf_current_A": -8.0e4,
                "curve_curve_min_dist": 0.0501,
                "curve_surface_min_dist": 0.067,
                "surface_vessel_min_dist": 0.082,
            },
            coil_length=2.1,
            accepted_iteration=1,
        )

        self.assertIs(payload["FINAL_FEASIBILITY_OK"], False)
        self.assertIs(payload["HARDWARE_CONSTRAINTS_OK"], False)
        self.assertEqual(
            payload["HARDWARE_CONSTRAINT_VIOLATIONS"],
            ["coil_length 2.100000 exceeds threshold 2.000000"],
        )

    def test_build_preserved_timeout_results_payload_backfills_missing_coil_length(
        self,
    ):
        module = load_single_stage_example_module()
        replay_config = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_test.nc",
            plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
        )
        run_dict = {
            "search_eval": {
                "total": 7.5e-4,
                "base_total": 7.4e-4,
            },
            "J": 7.5e-4,
            "intersecting": False,
            "surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"success": True},
        }

        payload = module.build_preserved_timeout_results_payload(
            replay_config=replay_config,
            preservation_kind="best_accepted",
            incumbent_stage="initial",
            run_dict=run_dict,
            objective_eval={"J_QS": 2.7e-4, "J_Boozer": 4.8e-7},
            field_error=3.5e-4,
            final_iota=0.14997,
            final_volume=0.09998,
            hardware_snapshot={
                "search_hardware_status": {"success": True, "violations": []},
                "artifact_hardware_status": {"success": True, "violations": []},
                "coil_length": None,
                "max_curvature": 19.8,
                "tf_current_A": -8.0e4,
                "curve_curve_min_dist": 0.0501,
                "curve_surface_min_dist": 0.067,
                "surface_vessel_min_dist": 0.082,
            },
            coil_length=1.83,
            accepted_iteration=1,
        )

        self.assertEqual(payload["COIL_LENGTH"], 1.83)

    def test_build_preserved_timeout_results_payload_frontier_uses_reference_metadata(
        self,
    ):
        module = load_single_stage_example_module()
        replay_config = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_test.nc",
            plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
            single_stage_goal_mode="frontier",
            single_stage_goal_mode_impl="frontier_tradeoff_score_v2",
            boozer_surface_target_volumes=(0.10,),
            frontier_iota_reference=0.15,
            frontier_iota_scale=0.05,
            frontier_volume_reference=0.10,
            frontier_volume_scale=0.01,
            frontier_qs_reference=2.0e-4,
            frontier_boozer_reference=1.0e-6,
            frontier_boozer_trust_threshold=1.0e-5,
            frontier_boozer_trust_penalty_scale=5.0e-5,
            frontier_effective_qs_weight=1.0,
            frontier_effective_boozer_weight=1.0,
            frontier_effective_iota_weight=1.0,
            frontier_effective_volume_weight=1.0,
            frontier_chebyshev_sharpness=18.0,
            frontier_epsilon_penalty_weight=9.0,
        )
        run_dict = {
            "search_eval": {
                "total": 7.5e-4,
                "base_total": 7.4e-4,
                "frontier_rank_total": 7.5e-4,
                "frontier_trust_ok": True,
                "frontier_boozer_trust_threshold": 1.0e-5,
                "frontier_boozer_trust_excess": 0.0,
                "frontier_boozer_trust_excess_ratio": 0.0,
                "frontier_boozer_trust_penalty_scale": 5.0e-5,
                "frontier_trust_penalty": 0.0,
                "J_volume": -0.2,
            },
            "J": 7.5e-4,
            "intersecting": False,
            "surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"success": True},
        }
        payload = module.build_preserved_timeout_results_payload(
            replay_config=replay_config,
            preservation_kind="best_feasible",
            incumbent_stage="initial",
            run_dict=run_dict,
            objective_eval={"J_QS": 2.7e-4, "J_Boozer": 4.8e-7},
            field_error=3.5e-4,
            final_iota=0.14997,
            final_volume=0.09998,
            hardware_snapshot={
                "search_hardware_status": {"success": True, "violations": []},
                "artifact_hardware_status": {"success": True, "violations": []},
                "max_curvature": 19.8,
                "tf_current_A": -8.0e4,
                "curve_curve_min_dist": 0.0496,
                "curve_surface_min_dist": 0.067,
                "surface_vessel_min_dist": 0.082,
            },
            coil_length=2.91,
            accepted_iteration=1,
        )

        self.assertIsNone(payload["TARGET_VOLUME"])
        self.assertIsNone(payload["TARGET_IOTA"])
        self.assertEqual(
            payload["SINGLE_STAGE_GOAL_MODE_IMPL"], "frontier_tradeoff_score_v2"
        )
        self.assertEqual(payload["BOOZER_SURFACE_TARGET_VOLUMES"], [0.10])
        self.assertEqual(payload["FRONTIER_REFERENCE_IOTA"], 0.15)
        self.assertEqual(payload["FRONTIER_REFERENCE_VOLUME"], 0.10)
        self.assertTrue(payload["FRONTIER_TRUST_OK"])
        self.assertEqual(payload["FRONTIER_BOOZER_TRUST_PENALTY_SCALE"], 5.0e-5)
        self.assertEqual(payload["FRONTIER_BOOZER_TRUST_EXCESS_RATIO"], 0.0)
        self.assertEqual(payload["FRONTIER_TRUST_PENALTY"], 0.0)
        self.assertEqual(payload["FRONTIER_CHEBYSHEV_SHARPNESS"], 18.0)
        self.assertEqual(payload["FRONTIER_EPSILON_PENALTY_WEIGHT"], 9.0)

    def test_current_preserved_timeout_replay_config_preserves_residue_objective_config(
        self,
    ):
        module = load_single_stage_example_module()
        replay_seed = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_test.nc",
            plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
        )
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            encoding="utf-8",
        ) as targets_file:
            targets_file.write('{"targets":[]}')
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            encoding="utf-8",
        ) as seeds_file:
            seeds_file.write('{"branch_seeds":[]}')
        self.addCleanup(Path(targets_file.name).unlink, missing_ok=True)
        self.addCleanup(Path(seeds_file.name).unlink, missing_ok=True)
        residue_args = SimpleNamespace(
            offspec_replay_debug_only=False,
            residue_objective_weight=0.25,
            residue_objective_targets_json=targets_file.name,
            residue_objective_seeds_json=seeds_file.name,
            residue_objective_axis_r=1.02,
            residue_objective_axis_z=0.01,
            residue_objective_poloidal_orientation=1,
            residue_objective_radial_label_scale=0.75,
            residue_objective_scale=0.5,
            residue_objective_r_satisfied=1.0e-5,
            residue_objective_local_difference_step=2.0e-6,
            residue_objective_rtol=1.0e-10,
            residue_objective_atol=1.0e-12,
            residue_objective_max_step=0.025,
            residue_objective_samples_per_full_torus=384,
            residue_objective_min_bphi_over_b=1.0e-7,
            residue_objective_newton_residual_tolerance=1.0e-10,
            residue_objective_winding_tolerance=1.0e-8,
            residue_objective_det_tolerance=2.0e-6,
            residue_objective_max_newton_iterations=16,
            residue_objective_max_newton_step_norm=0.025,
        )
        residue_objective = SimpleNamespace(
            target_manifest_id="sha256:test-targets",
            validation_id="validation-artifact",
        )

        with (
            patch.object(module, "PRESERVED_TIMEOUT_REPLAY_CONFIG", replay_seed),
            patch.object(module, "args", residue_args, create=True),
            patch.object(
                module,
                "JResidueObjective",
                residue_objective,
                create=True,
            ),
        ):
            replay_config = module.current_preserved_timeout_replay_config()

        payload = module.residue_objective_replay_config_payload(
            replay_config.residue_objective_replay_config
        )
        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["weight"], 0.25)
        self.assertEqual(payload["target_manifest_id"], "sha256:test-targets")
        self.assertEqual(payload["validation_id"], "validation-artifact")
        self.assertEqual(
            payload["targets_sha256"],
            module._optional_file_sha256(targets_file.name),
        )
        self.assertEqual(
            payload["seeds_sha256"],
            module._optional_file_sha256(seeds_file.name),
        )
        self.assertEqual(payload["local_difference_step"], 2.0e-6)

    def test_current_preserved_timeout_replay_config_preserves_cws_shape_metadata(
        self,
    ):
        module = load_single_stage_example_module()
        replay_seed = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_test.nc",
            plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
            winding_surface_free_mpol=1,
            winding_surface_free_ntor=1,
        )

        with (
            patch.object(module, "PRESERVED_TIMEOUT_REPLAY_CONFIG", replay_seed),
            patch.object(
                module,
                "winding_surface_free_dof_names",
                ("rc(0,1)", "zs(0,1)"),
                create=True,
            ),
            patch.object(
                module,
                "surf_coils",
                SimpleNamespace(
                    mpol=1,
                    ntor=1,
                    get_rc=lambda m, n: 0.920 if (m, n) == (0, 0) else 0.142,
                ),
                create=True,
            ),
            patch.object(
                module,
                "winding_surface_shape_requested",
                True,
                create=True,
            ),
        ):
            replay_config = module.current_preserved_timeout_replay_config()

        self.assertEqual(replay_config.winding_surface_free_mpol, 1)
        self.assertEqual(replay_config.winding_surface_free_ntor, 1)
        self.assertEqual(
            replay_config.winding_surface_free_dof_names,
            ("rc(0,1)", "zs(0,1)"),
        )
        self.assertEqual(replay_config.coil_winding_surface_mpol, 1)
        self.assertEqual(replay_config.coil_winding_surface_ntor, 1)
        self.assertEqual(replay_config.coil_winding_surface_major_radius_m, 0.920)
        self.assertEqual(
            replay_config.banana_cws_embedded_winding_minor_radius_m,
            0.142,
        )
        self.assertTrue(replay_config.banana_cws_reembedded_on_live_surface)

    def test_current_preserved_timeout_replay_config_keeps_seed_cws_radii_without_reembed(
        self,
    ):
        module = load_single_stage_example_module()
        replay_seed = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_test.nc",
            plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
            winding_surface_free_mpol=0,
            winding_surface_free_ntor=0,
            coil_winding_surface_major_radius_m=0.976,
            banana_cws_embedded_winding_minor_radius_m=0.210,
            banana_cws_reembedded_on_live_surface=False,
        )

        with (
            patch.object(module, "PRESERVED_TIMEOUT_REPLAY_CONFIG", replay_seed),
            patch.object(
                module,
                "surf_coils",
                SimpleNamespace(
                    mpol=1,
                    ntor=0,
                    get_rc=lambda m, n: 0.950 if (m, n) == (0, 0) else 0.142,
                ),
                create=True,
            ),
            patch.object(
                module,
                "winding_surface_shape_requested",
                False,
                create=True,
            ),
        ):
            replay_config = module.current_preserved_timeout_replay_config()

        self.assertEqual(replay_config.coil_winding_surface_major_radius_m, 0.976)
        self.assertEqual(
            replay_config.banana_cws_embedded_winding_minor_radius_m,
            0.210,
        )
        self.assertFalse(replay_config.banana_cws_reembedded_on_live_surface)

    def test_current_preserved_timeout_replay_config_round_trips_frontier_scalarization_overrides(
        self,
    ):
        module = load_single_stage_example_module()
        replay_seed = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_test.nc",
            plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
            single_stage_goal_mode="frontier",
            single_stage_goal_mode_impl="frontier_tradeoff_score_v2",
        )
        frontier_goal_config = make_frontier_goal_config(
            module,
            scalarization_type="achievement_chebyshev_sweep_v1",
            chebyshev_rho=0.02,
            chebyshev_sharpness=18.0,
            epsilon_penalty_weight=9.0,
        )
        run_dict = {
            "search_eval": {
                "total": 7.5e-4,
                "base_total": 7.4e-4,
                "frontier_rank_total": 7.5e-4,
            },
            "J": 7.5e-4,
            "intersecting": False,
            "surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"success": True},
        }

        with (
            patch.object(module, "PRESERVED_TIMEOUT_REPLAY_CONFIG", replay_seed),
            patch.object(
                module,
                "FRONTIER_GOAL_CONFIG",
                frontier_goal_config,
            ),
        ):
            replay_config = module.current_preserved_timeout_replay_config()

        payload = module.build_preserved_timeout_results_payload(
            replay_config=replay_config,
            preservation_kind="best_feasible",
            incumbent_stage="initial",
            run_dict=run_dict,
            objective_eval={"J_QS": 2.7e-4, "J_Boozer": 4.8e-7},
            field_error=3.5e-4,
            final_iota=0.14997,
            final_volume=0.09998,
            hardware_snapshot={
                "search_hardware_status": {"success": True, "violations": []},
                "artifact_hardware_status": {"success": True, "violations": []},
                "max_curvature": 19.8,
                "tf_current_A": -8.0e4,
                "curve_curve_min_dist": 0.0496,
                "curve_surface_min_dist": 0.067,
                "surface_vessel_min_dist": 0.082,
            },
            coil_length=2.91,
            accepted_iteration=1,
        )

        self.assertEqual(
            replay_config.frontier_scalarization_type, "achievement_chebyshev_sweep_v1"
        )
        self.assertEqual(replay_config.frontier_chebyshev_rho, 0.02)
        self.assertEqual(replay_config.frontier_chebyshev_sharpness, 18.0)
        self.assertEqual(replay_config.frontier_epsilon_penalty_weight, 9.0)
        self.assertEqual(
            payload["FRONTIER_SCALARIZATION_TYPE"], "achievement_chebyshev_sweep_v1"
        )
        self.assertEqual(payload["FRONTIER_CHEBYSHEV_RHO"], 0.02)
        self.assertEqual(payload["FRONTIER_CHEBYSHEV_SHARPNESS"], 18.0)
        self.assertEqual(payload["FRONTIER_EPSILON_PENALTY_WEIGHT"], 9.0)

    def test_current_preserved_timeout_replay_config_preserves_offspec_resume_role(
        self,
    ):
        module = load_single_stage_example_module()
        replay_seed = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_test.nc",
            plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
        )

        with (
            patch.object(
                module,
                "PRESERVED_TIMEOUT_REPLAY_CONFIG",
                replay_seed,
            ),
            patch.object(
                module,
                "stage2_bs_path",
                "/resume/biot_savart_opt.json",
                create=True,
            ),
            patch.object(
                module,
                "seed_artifact_role",
                "single_stage_resume",
                create=True,
            ),
            patch.object(
                module,
                "args",
                SimpleNamespace(offspec_replay_debug_only=True),
                create=True,
            ),
        ):
            replay_config = module.current_preserved_timeout_replay_config()

        self.assertEqual(replay_config.seed_artifact_role, "single_stage_resume")
        self.assertTrue(replay_config.offspec_replay_debug_only)
        self.assertEqual(
            replay_config.single_stage_resume_bs_path,
            "/resume/biot_savart_opt.json",
        )

    def test_current_preserved_timeout_replay_config_preserves_strict_vacuum_fields(
        self,
    ):
        module = load_single_stage_example_module()
        replay_seed = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_test.nc",
            plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
            strict_vacuum_current=True,
            strict_vacuum_seed_lineage="recent_stage1_candidate",
            stage1_candidate_id="s01_3240f0",
            strict_vacuum_source_current_group_projection="tf_banana_only",
            strict_vacuum_command_validation={"passed": True},
            strict_vacuum_seed_input_validation={"passed": True},
            finite_current_mode=None,
            effective_current_mode="vacuum",
            boozer_current_convention=None,
            plasma_current_A=0.0,
            boozer_I=0.0,
            num_proxy_coils=0,
            num_vf_coils=0,
            proxy_plasma_current_A=0.0,
            vf_current_A=0.0,
        )

        class _Current:
            dof_names = ()

            def get_value(self):
                return -15910.0

        BoozerSurface = type("BoozerSurface", (), {})
        BoozerSurface.__module__ = "simsopt.geo.boozersurface"
        surface_data = [{"boozer_surface": BoozerSurface()}]
        banana_current_state = module.SingleStageBananaCurrentState(
            mode="shared",
            currents=(_Current(),),
            seed_currents_A=(-15910.0,),
        )
        with patch.object(module, "PRESERVED_TIMEOUT_REPLAY_CONFIG", replay_seed):
            with (
                patch.object(module, "stage2_tf_current_A", -80000.0, create=True),
                patch.object(
                    module,
                    "banana_current_state",
                    banana_current_state,
                    create=True,
                ),
            ):
                replay_config = module.current_preserved_timeout_replay_config()
                manifest = module.current_boozer_state_interchange_manifest(
                    surface_data
                )

        self.assertTrue(replay_config.strict_vacuum_current)
        self.assertEqual(
            replay_config.strict_vacuum_seed_lineage,
            "recent_stage1_candidate",
        )
        self.assertEqual(replay_config.stage1_candidate_id, "s01_3240f0")
        self.assertEqual(
            replay_config.strict_vacuum_source_current_group_projection,
            "tf_banana_only",
        )
        self.assertEqual(replay_config.effective_current_mode, "vacuum")
        self.assertEqual(replay_config.boozer_I, 0.0)
        self.assertEqual(replay_config.num_proxy_coils, 0)
        self.assertIsNotNone(manifest)
        self.assertTrue(manifest["baseline_replayable"])

    def test_current_boozer_state_interchange_manifest_rejects_positive_current_state(
        self,
    ):
        module = load_single_stage_example_module()
        replay_seed = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_test.nc",
            plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
            strict_vacuum_current=True,
            strict_vacuum_seed_lineage="legacy_control",
        )

        class _Current:
            dof_names = ()

            def get_value(self):
                return 15910.0

        BoozerSurface = type("BoozerSurface", (), {})
        BoozerSurface.__module__ = "simsopt.geo.boozersurface"
        surface_data = [{"boozer_surface": BoozerSurface()}]
        banana_current_state = module.SingleStageBananaCurrentState(
            mode="shared",
            currents=(_Current(),),
            seed_currents_A=(15910.0,),
        )

        with (
            patch.object(module, "PRESERVED_TIMEOUT_REPLAY_CONFIG", replay_seed),
            patch.object(module, "stage2_tf_current_A", -80000.0, create=True),
            patch.object(
                module,
                "banana_current_state",
                banana_current_state,
                create=True,
            ),
            self.assertRaisesRegex(ValueError, "signed_banana_current_negative"),
        ):
            module.current_boozer_state_interchange_manifest(surface_data)

    def test_build_best_feasible_results_summary_emits_schema_backed_hardware_fields(
        self,
    ):
        module = load_single_stage_example_module()
        run_dict = {
            "best_feasible_incumbent": {"surface_state": "saved"},
            "best_feasible_stage": "accepted",
            "J": 7.5e-4,
            "dJ": np.array([1.0, -1.0]),
            "search_eval": {
                "total": 7.5e-4,
                "physics_total": 7.4e-4,
                "J_QS": 2.7e-4,
                "J_Boozer": 4.8e-7,
                "frontier_rank_total": None,
                "frontier_trust_ok": True,
            },
            "surface_status": {
                "iotas": [0.14997],
                "volumes": [0.09998],
                "success": True,
                "self_intersections": [False],
            },
            "topology_gate_status": {
                "success": True,
                "state": "pass",
                "evaluation_error": None,
                "transport_diagnostics": None,
            },
        }
        hardware_snapshot = {
            "artifact_hardware_status": {
                "success": False,
                "violations": ["coil_length 2.100000 exceeds threshold 2.000000"],
            },
            "curve_curve_min_dist": 0.0501,
            "curve_surface_min_dist": 0.067,
            "surface_vessel_min_dist": 0.082,
            "poloidal_extent_rad": 0.0,
            "poloidal_extent_threshold_rad": module.POLOIDAL_EXTENT_HALF_WIDTH_RAD,
            "max_curvature": 19.8,
            "coil_length": 2.1,
            "length_target": 1.9,
            "tf_current_A": -8.0e4,
            "tf_current_limit_A": 8.0e4,
            "banana_current_A": 1.4e4,
            "banana_current_max_A": 1.6e4,
        }

        with (
            patch.object(
                module,
                "snapshot_single_stage_incumbent_state",
                return_value={"surface_state": "current"},
            ),
            patch.object(
                module,
                "restore_single_stage_incumbent_state",
            ),
            patch.object(
                module,
                "evaluate_single_stage_hardware_snapshot",
                return_value=hardware_snapshot,
            ),
            patch.object(
                module,
                "build_topology_gate_diagnostics",
                return_value={"kind": "gate", "outcome": "pass"},
            ),
        ):
            summary = module.build_best_feasible_results_summary(
                run_dict,
                curve_curve_distance_obj=object(),
                curve_surface_distance_obj=object(),
                banana_curve=self.inboard_midplane_curve(),
                curvelength_obj=SimpleNamespace(J=lambda: 2.1),
                cc_dist=0.05,
                cs_dist=0.015,
                ss_dist=0.04,
                curvature_threshold=100.0,
                length_target=1.7,
                tf_current_A=-8.0e4,
                banana_coils=[
                    SimpleNamespace(current=SimpleNamespace(get_value=lambda: 1.4e4))
                ],
                banana_current_max_A=1.6e4,
                outer_surface=object(),
            )

        self.assertTrue(summary["BEST_FEASIBLE_AVAILABLE"])
        self.assertEqual(summary["BEST_FEASIBLE_STAGE"], "accepted")
        self.assertEqual(summary["BEST_FEASIBLE_CURVE_CURVE_MIN_DIST"], 0.0501)
        self.assertEqual(summary["BEST_FEASIBLE_CURVE_SURFACE_MIN_DIST"], 0.067)
        self.assertEqual(summary["BEST_FEASIBLE_POLOIDAL_EXTENT_RAD"], 0.0)
        self.assertEqual(
            summary["BEST_FEASIBLE_POLOIDAL_EXTENT_THRESHOLD_RAD"],
            module.POLOIDAL_EXTENT_HALF_WIDTH_RAD,
        )
        self.assertEqual(summary["BEST_FEASIBLE_MAX_CURVATURE"], 19.8)
        self.assertEqual(summary["BEST_FEASIBLE_COIL_LENGTH"], 2.1)
        self.assertEqual(summary["BEST_FEASIBLE_LENGTH_TARGET"], 1.9)
        self.assertEqual(summary["BEST_FEASIBLE_TF_CURRENT_A"], -8.0e4)
        self.assertEqual(summary["BEST_FEASIBLE_TF_CURRENT_LIMIT_A"], 8.0e4)
        self.assertEqual(summary["BEST_FEASIBLE_BANANA_CURRENT_A"], 1.4e4)
        self.assertEqual(summary["BEST_FEASIBLE_BANANA_CURRENT_MAX_A"], 1.6e4)
        self.assertFalse(summary["BEST_FEASIBLE_HARDWARE_CONSTRAINTS_OK"])
        self.assertEqual(
            summary["BEST_FEASIBLE_HARDWARE_CONSTRAINT_VIOLATIONS"],
            ["coil_length 2.100000 exceeds threshold 2.000000"],
        )

    def test_build_best_feasible_results_summary_uses_diagnostic_incumbent_snapshot(
        self,
    ):
        module = load_single_stage_example_module()
        compact_eval = {
            "total": 7.5e-4,
            "physics_total": 7.4e-4,
            "grad": np.array([1.0, -1.0]),
            "surface_weights": np.array([1.0]),
            "diagnostics_included": False,
        }
        rich_eval = diagnostic_search_eval_payload(
            {
                **compact_eval,
                "J_QS": 2.7e-4,
                "J_Boozer": 4.8e-7,
            }
        )
        run_dict = {
            "accepted_x": np.array([1.0, 2.0]),
            "surface_state": {"sdofs": [np.array([1.0])], "iota": [0.15], "G": [1.0]},
            "J": compact_eval["total"],
            "dJ": np.array([1.0, -1.0]),
            "search_eval": compact_eval,
            "surface_status": {
                "iotas": [0.14997],
                "volumes": [0.09998],
                "success": True,
                "self_intersections": [False],
            },
            "search_surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {
                "enabled": False,
                "success": True,
                "state": "pass",
                "evaluation_error": None,
                "transport_diagnostics": None,
            },
            "intersecting": False,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
        }
        hardware_snapshot = {
            "artifact_hardware_status": {"success": True, "violations": []},
            "curve_curve_min_dist": 0.0501,
            "curve_surface_min_dist": 0.067,
            "surface_vessel_min_dist": 0.082,
            "poloidal_extent_rad": 0.0,
            "poloidal_extent_threshold_rad": module.POLOIDAL_EXTENT_HALF_WIDTH_RAD,
            "max_curvature": 19.8,
            "coil_length": 2.1,
            "length_target": 1.9,
            "tf_current_A": -8.0e4,
            "tf_current_limit_A": 8.0e4,
            "banana_current_A": 1.4e4,
            "banana_current_max_A": 1.6e4,
        }

        with patch.object(
            module,
            "evaluate_search_objective",
            return_value=rich_eval,
        ) as evaluate_mock:
            self.assertTrue(
                module.maybe_update_best_feasible_incumbent(run_dict, "initial")
            )

        self.assertTrue(evaluate_mock.call_args.kwargs["include_diagnostics"])
        self.assertNotIn("J_QS", run_dict["search_eval"])

        with (
            patch.object(
                module,
                "evaluate_single_stage_hardware_snapshot",
                return_value=hardware_snapshot,
            ),
            patch.object(
                module,
                "build_topology_gate_diagnostics",
                return_value={"kind": "gate", "outcome": "pass"},
            ),
        ):
            summary = module.build_best_feasible_results_summary(
                run_dict,
                curve_curve_distance_obj=object(),
                curve_surface_distance_obj=object(),
                banana_curve=self.inboard_midplane_curve(),
                curvelength_obj=SimpleNamespace(J=lambda: 2.1),
                cc_dist=0.05,
                cs_dist=0.015,
                ss_dist=0.04,
                curvature_threshold=100.0,
                length_target=1.7,
                tf_current_A=-8.0e4,
                banana_coils=[
                    SimpleNamespace(current=SimpleNamespace(get_value=lambda: 1.4e4))
                ],
                banana_current_max_A=1.6e4,
                outer_surface=object(),
            )

        self.assertTrue(summary["BEST_FEASIBLE_AVAILABLE"])
        self.assertAlmostEqual(summary["BEST_FEASIBLE_QA_OBJECTIVE"], 2.7e-4)
        self.assertAlmostEqual(summary["BEST_FEASIBLE_BOOZER_OBJECTIVE"], 4.8e-7)

    def test_validate_boozer_stage_refinement_args_rejects_unsupported_scope(self):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            boozer_stage_refinement=True,
            constraint_method="alm",
            num_surfaces=1,
            basin_hops=0,
            boozer_stage="initial",
            refinement_boozer_stage="final",
            refinement_maxiter=20,
            refinement_chunk_maxiter=10,
            refinement_max_stalled_chunks=2,
        )

        with self.assertRaisesRegex(ValueError, "--constraint-method=penalty"):
            module.validate_boozer_stage_refinement_args(args, constraint_weight=1.0)

        args.constraint_method = "penalty"
        args.num_surfaces = 2
        with self.assertRaisesRegex(ValueError, "--num-surfaces=1"):
            module.validate_boozer_stage_refinement_args(args, constraint_weight=1.0)

        args.num_surfaces = 1
        args.refinement_chunk_maxiter = 0
        with self.assertRaisesRegex(
            ValueError, "--refinement-chunk-maxiter must be positive"
        ):
            module.validate_boozer_stage_refinement_args(args, constraint_weight=1.0)

    def test_refinement_improves_phase1_metric_uses_phase1_stage_basis(self):
        module = load_single_stage_example_module()
        run_dict = {
            "accepted_x": np.array([1.0, 2.0]),
            "surface_state": {"sdofs": [np.array([1.0])], "iota": [0.15], "G": [1.0]},
            "J": 6.0,
            "dJ": np.array([1.0, -1.0]),
            "search_eval": {"total": 6.0, "surface_weights": np.array([1.0])},
            "surface_status": {"success": True},
            "search_surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"enabled": False, "success": True},
            "intersecting": False,
            "accepted_iterations": 0,
        }
        refinement_incumbent = module.snapshot_single_stage_incumbent_state(run_dict)
        rebuild_calls = []

        def fake_rebuild(stage_name):
            rebuild_calls.append(stage_name)

        with patch.object(
            module,
            "refresh_accepted_search_state",
            autospec=True,
            side_effect=lambda current_run_dict, stage_name: (
                2.5 if stage_name == "initial" else 9.0
            ),
        ):
            refinement_metric, refinement_improved = (
                module.refinement_improves_phase1_metric(
                    3.0,
                    "initial",
                    run_dict,
                    refinement_incumbent,
                    fake_rebuild,
                )
            )

        self.assertEqual(refinement_metric, 2.5)
        self.assertTrue(refinement_improved)
        self.assertEqual(rebuild_calls, ["initial"])

    def test_reported_boozer_stage_follows_saved_final_source(self):
        module = load_single_stage_example_module()
        self.assertEqual(module.reported_boozer_stage("initial", "final"), "final")
        self.assertEqual(module.reported_boozer_stage("initial", None), "initial")

    def test_run_chunked_refinement_aborts_after_stalled_chunks_without_improvement(
        self,
    ):
        module = load_single_stage_example_module()
        phase1_incumbent = SimpleNamespace(name="phase1")
        stalled_incumbent = SimpleNamespace(name="stalled")
        chunk_results = [
            (
                SimpleNamespace(nit=5, success=False, message="chunk1"),
                stalled_incumbent,
            ),
            (
                SimpleNamespace(nit=4, success=False, message="chunk2"),
                stalled_incumbent,
            ),
        ]

        with (
            patch.object(module, "run_refinement_chunk", side_effect=chunk_results),
            patch.object(
                module,
                "refinement_improves_phase1_metric",
                side_effect=[(6.0, False), (6.0, False)],
            ),
        ):
            result = module.run_chunked_refinement(
                {},
                phase1_incumbent,
                5.0,
                "initial",
                "final",
                lambda stage_name: None,
                20,
                10,
                2,
                300,
                1e-9,
                1e-9,
            )

        self.assertIsNone(result["best_incumbent"])
        self.assertEqual(result["iterations"], 9)
        self.assertEqual(result["chunks"], 2)
        self.assertEqual(result["abort_reason"], "stalled_without_improvement")
        self.assertEqual(
            result["termination_message"], "chunk2; stalled_without_improvement"
        )

    def test_run_chunked_refinement_keeps_best_improvement_after_later_stall(self):
        module = load_single_stage_example_module()
        phase1_incumbent = SimpleNamespace(name="phase1")
        improved_incumbent = SimpleNamespace(name="improved")
        chunk_results = [
            (
                SimpleNamespace(nit=6, success=False, message="chunk1"),
                improved_incumbent,
            ),
            (
                SimpleNamespace(nit=3, success=False, message="chunk2"),
                improved_incumbent,
            ),
        ]

        with (
            patch.object(module, "run_refinement_chunk", side_effect=chunk_results),
            patch.object(
                module,
                "refinement_improves_phase1_metric",
                side_effect=[(4.0, True), (4.0, False)],
            ),
        ):
            result = module.run_chunked_refinement(
                {},
                phase1_incumbent,
                5.0,
                "initial",
                "final",
                lambda stage_name: None,
                20,
                10,
                1,
                300,
                1e-9,
                1e-9,
            )

        self.assertIs(result["best_incumbent"], improved_incumbent)
        self.assertEqual(result["best_metric"], 4.0)
        self.assertEqual(result["iterations"], 9)
        self.assertEqual(result["chunks"], 2)
        self.assertEqual(result["abort_reason"], "stalled_after_improvement")
        self.assertEqual(
            result["termination_message"], "chunk2; stalled_after_improvement"
        )

    def test_run_chunked_refinement_reports_budget_exhaustion_after_improvement(self):
        module = load_single_stage_example_module()
        phase1_incumbent = SimpleNamespace(name="phase1")
        improved_incumbent = SimpleNamespace(name="improved")

        with (
            patch.object(
                module,
                "run_refinement_chunk",
                return_value=(
                    SimpleNamespace(nit=5, success=False, message="chunk1"),
                    improved_incumbent,
                ),
            ),
            patch.object(
                module,
                "refinement_improves_phase1_metric",
                return_value=(4.0, True),
            ),
        ):
            result = module.run_chunked_refinement(
                {},
                phase1_incumbent,
                5.0,
                "initial",
                "final",
                lambda stage_name: None,
                5,
                5,
                2,
                300,
                1e-9,
                1e-9,
            )

        self.assertIs(result["best_incumbent"], improved_incumbent)
        self.assertEqual(result["best_metric"], 4.0)
        self.assertEqual(result["abort_reason"], "budget_exhausted_after_improvement")
        self.assertEqual(
            result["termination_message"], "chunk1; budget_exhausted_after_improvement"
        )

    def test_summarize_refinement_result_uses_refinement_status(self):
        module = load_single_stage_example_module()
        accepted_x = np.array([1.0, 2.0])
        refinement_result = {
            "termination_message": "stalled_without_improvement",
            "success": False,
        }

        termination_message, optimizer_success, result = (
            module.summarize_refinement_result(
                refinement_result,
                total_iterations=13,
                accepted_x=accepted_x,
            )
        )

        self.assertEqual(termination_message, "stalled_without_improvement")
        self.assertFalse(optimizer_success)
        self.assertEqual(result.nit, 13)
        self.assertEqual(result.message, "stalled_without_improvement")
        self.assertFalse(result.success)
        np.testing.assert_array_equal(result.x, accepted_x)

    def test_fun_rejects_candidate_on_hardware_constraint_failure(self):
        module, J_out, dJ_out, last_dJ, restore_mock = (
            self._run_fun_with_hardware_violation(
                hardware_search_mode="hard",
            )
        )

        self.assertEqual(J_out, 24.0)
        np.testing.assert_array_equal(dJ_out, last_dJ)
        restore_mock.assert_called_once()
        self.assertFalse(module.run_dict["trial_hardware_status"]["success"])
        self.assertIsNone(module.run_dict["accepted_hardware_status"])

    def test_evaluate_search_step_curvature_precheck_rejects_before_boozer(self):
        module = self.load_module()
        last_dJ = np.array([1.0, -2.0])

        class _JF:
            x = np.zeros(2)

        class _Curve:
            def kappa(self):
                return np.array([106.0])

        module.CONSTRAINT_METHOD = "penalty"
        module.CURVATURE_THRESHOLD = 100.0
        module.CURVATURE_TRAVERSAL_BAND = 0.05
        module.CURVATURE_TRAVERSAL_EVAL_BUDGET = 2
        module.MULTISURFACE_RAMP_ITERATIONS = 0
        module.INNER_SURFACE_INITIAL_WEIGHT = 1.0
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.JF = _JF()
        module.banana_curve = _Curve()
        module.surface_data = [{"boozer_surface": SimpleNamespace(surface=object())}]
        module.run_dict = {
            "x_prev": np.zeros(2),
            "lscount": 0,
            "surface_state": {"sdofs": [], "iota": [], "G": []},
            "accepted_x": np.zeros(2),
            "J": 5.0,
            "dJ": last_dJ.copy(),
            "accepted_iterations": 0,
            "trial_hardware_status": None,
            "invalid_state_rejects_total": 0,
            "topology_gate_rejects": 0,
            "hardware_rejects": 0,
            "surface_solve_rejects": 0,
            "frontier_trust_rejects": 0,
            "curvature_precheck_rejects": 0,
            "curvature_overcap_boozer_evals": 0,
            "curvature_overcap_boozer_evals_this_iteration": 0,
        }

        with (
            patch.object(
                module,
                "solve_surface_stack_at_dofs",
                side_effect=AssertionError("Boozer solve should not run"),
            ),
            patch.object(module, "restore_surface_states") as restore_mock,
        ):
            evaluation = module.evaluate_search_step(np.ones(2))

        self.assertEqual(evaluation["total"], 10.0)
        np.testing.assert_array_equal(evaluation["grad"], last_dJ)
        self.assertEqual(module.run_dict["invalid_state_rejects_total"], 1)
        self.assertEqual(module.run_dict["curvature_precheck_rejects"], 1)
        self.assertEqual(module.run_dict["curvature_overcap_boozer_evals"], 0)
        self.assertEqual(
            module.run_dict["curvature_traversal_status"]["reason"],
            "far_invalid_curvature",
        )
        restore_mock.assert_called_once()
        payload = module.search_step_metrics_payload(module.run_dict)
        self.assertEqual(payload["SEARCH_STEP_REJECTED_BEFORE_SURFACE_SOLVE"], 1)
        self.assertEqual(payload["SEARCH_STEP_CURVATURE_PRECHECK_REJECTS"], 1)

    def test_curvature_traversal_precheck_tracks_overcap_budget(self):
        module = self.load_module()

        class _JF:
            x = np.zeros(2)

        class _Curve:
            def kappa(self):
                return np.array([104.0])

        module.CONSTRAINT_METHOD = "penalty"
        module.CURVATURE_THRESHOLD = 100.0
        module.CURVATURE_TRAVERSAL_BAND = 0.05
        module.CURVATURE_TRAVERSAL_EVAL_BUDGET = 1
        module.JF = _JF()
        module.banana_curve = _Curve()
        module.run_dict = {
            "curvature_overcap_boozer_evals": 0,
            "curvature_overcap_boozer_evals_this_iteration": 0,
        }
        metrics = module.new_search_step_metrics()

        first_status = module.evaluate_curvature_traversal_precheck(
            np.array([1.0, 2.0]),
            metrics,
        )
        second_status = module.evaluate_curvature_traversal_precheck(
            np.array([3.0, 4.0]),
            metrics,
        )

        self.assertTrue(first_status["allow_boozer_eval"])
        self.assertEqual(first_status["reason"], "within_traversal_band")
        self.assertFalse(second_status["allow_boozer_eval"])
        self.assertEqual(
            second_status["reason"], "curvature_traversal_budget_exhausted"
        )
        self.assertEqual(module.run_dict["curvature_overcap_boozer_evals"], 1)
        self.assertEqual(
            module.run_dict["curvature_overcap_boozer_evals_this_iteration"],
            1,
        )
        self.assertEqual(metrics["curvature_overcap_boozer_evals"], 1)

    def test_fun_allows_inside_band_curvature_only_violation_in_hard_mode(self):
        module, J_out, dJ_out, _last_dJ, restore_mock = (
            self._run_fun_with_hardware_violation(
                hardware_search_mode="hard",
                curvature_traversal_band=0.05,
                curvature_traversal_eval_budget=1,
                search_hardware_values=(0.0, 0.0, 0.25),
            )
        )

        self.assertEqual(J_out, 7.0)
        np.testing.assert_array_equal(dJ_out, np.arange(3, dtype=float))
        restore_mock.assert_not_called()
        hardware_status = module.run_dict["trial_hardware_status"]

        self.assertTrue(hardware_status["success"])
        self.assertTrue(hardware_status["curvature_traversal_allowed"])
        self.assertTrue(hardware_status["constraints"]["max_curvature"]["success"])
        self.assertTrue(hardware_status["allowed_traversal_status"]["success"])
        self.assertTrue(hardware_status["curvature_traversal_original_violations"])
        violation_ratios = module._hardware_violation_ratios(hardware_status)
        self.assertEqual(max(violation_ratios.values()), 0.0)
        self.assertEqual(module.run_dict["hardware_rejects"], 0)
        self.assertEqual(module.run_dict["invalid_state_rejects_total"], 0)
        self.assertEqual(module.run_dict["curvature_overcap_boozer_evals"], 1)

    def test_fun_warns_only_on_hardware_constraint_failure_in_warn_mode(self):
        module, J_out, dJ_out, _last_dJ, restore_mock = (
            self._run_fun_with_hardware_violation(
                hardware_search_mode="warn",
            )
        )

        self.assertEqual(J_out, 7.0)
        np.testing.assert_array_equal(dJ_out, np.arange(3, dtype=float))
        restore_mock.assert_not_called()
        self.assertFalse(module.run_dict["trial_hardware_status"]["success"])
        self.assertIsNone(module.run_dict["accepted_hardware_status"])

    def test_fun_rejects_hardware_violation_in_adaptive_mode_when_gate_not_relaxed(
        self,
    ):
        module, J_out, dJ_out, _last_dJ, restore_mock = (
            self._run_fun_with_hardware_violation(
                hardware_search_mode="adaptive",
                hardware_search_soft_iterations=1,
                accepted_iterations=0,
            )
        )

        self.assertEqual(J_out, 24.0)
        np.testing.assert_array_equal(dJ_out, np.array([1.0, -1.0, 2.0]))
        restore_mock.assert_called_once()
        self.assertFalse(module.run_dict["trial_hardware_status"]["success"])
        self.assertIsNone(module.run_dict["accepted_hardware_status"])

    def test_fun_warns_in_adaptive_mode_only_while_gate_is_relaxed(self):
        module, J_out, dJ_out, _last_dJ, restore_mock = (
            self._run_fun_with_hardware_violation(
                hardware_search_mode="adaptive",
                hardware_search_soft_iterations=1,
                accepted_iterations=0,
            )
        )

        module.MULTISURFACE_RAMP_ITERATIONS = 5
        module.INNER_SURFACE_INITIAL_WEIGHT = 0.0
        module.run_dict["accepted_iterations"] = 0

        with (
            patch.object(module, "restore_surface_states") as adaptive_restore_mock,
            patch.object(
                module,
                "solve_surface_stack_at_dofs",
                return_value={
                    "success": True,
                    "solve_success": [True, True],
                    "self_intersections": [False, False],
                    "volumes_ordered": True,
                    "gap_ok": True,
                    "nesting_ok": True,
                    "adjacent_gaps": [0.1],
                    "outer_vessel_gap": 0.05,
                    "bad_nesting_phis": [],
                },
            ),
            patch.object(
                module,
                "evaluate_total_objective",
                return_value={
                    "total": 7.0,
                    "grad": np.arange(3, dtype=float),
                    "surface_weights": np.array([0.0, 1.0]),
                    "J_QS": 0.0,
                    "dJ_QS": np.zeros(3),
                    "J_Boozer": 0.0,
                    "dJ_Boozer": np.zeros(3),
                    "J_iota": 0.0,
                    "dJ_iota": np.zeros(3),
                    "J_curvature": 0.0,
                    "dJ_curvature": np.zeros(3),
                    **search_hardware_penalty_payload((0.25, 0.0, 0.0)),
                },
            ),
            patch.object(
                module,
                "evaluate_surface_stack",
                return_value={
                    "success": True,
                    "solve_success": [True, True],
                    "self_intersections": [False, False],
                    "volumes_ordered": True,
                    "gap_ok": True,
                    "nesting_ok": True,
                    "adjacent_gaps": [0.1],
                    "outer_vessel_gap": 0.05,
                    "bad_nesting_phis": [],
                },
            ),
            patch.object(
                module,
                "evaluate_single_stage_hardware_snapshot",
                return_value=topology_hardware_snapshot(),
            ),
            patch.object(
                module,
                "compute_surface_field_metrics",
                return_value=(0.0, 0.0),
            ),
            patch.object(
                module,
                "maybe_write_best_hardware_near_miss_trial_artifacts",
                return_value=False,
            ),
        ):
            J_out, dJ_out = module.fun(np.ones(3))

        self.assertEqual(J_out, 7.0)
        np.testing.assert_array_equal(dJ_out, np.arange(3, dtype=float))
        restore_mock.assert_called_once()
        adaptive_restore_mock.assert_not_called()
        self.assertFalse(module.run_dict["trial_hardware_status"]["success"])

    def test_callback_records_accepted_invalid_hardware_status_after_warn_mode_step(
        self,
    ):
        module, J_out, _dJ_out, _last_dJ, restore_mock = (
            self._run_fun_with_hardware_violation(
                hardware_search_mode="warn",
            )
        )

        self.assertEqual(J_out, 7.0)
        restore_mock.assert_not_called()

        class _Surface:
            nfp = 5

            def __init__(self):
                self.x = np.array([0.1])

            def volume(self):
                return 1.0

            def gamma(self):
                return np.array([[[0.0, 0.0, 0.0]]])

            def unitnormal(self):
                return np.array([[[1.0, 0.0, 0.0]]])

            def save(self, path):
                self._saved_path = path

        class _ScalarObjective:
            def __init__(self, value):
                self._value = value

            def J(self):
                return self._value

            def dJ(self):
                return np.array([self._value, -self._value, 0.0])

        class _DistanceObjective(_ScalarObjective):
            def __init__(self, value, min_distance):
                super().__init__(value)
                self._min_distance = min_distance

            def shortest_distance(self):
                return self._min_distance

        class _Curve:
            def gamma(self):
                return np.array([[1.0, 0.0, 0.0]])

            def kappa(self):
                return np.array([41.0])

        class _CurveLength:
            def J(self):
                return 1.7

        class _BS:
            def set_points(self, pts):
                self._points = pts

            def B(self):
                return np.array([[1.0, 0.0, 0.0]])

            def save(self, path):
                self._saved_path = path

        surface = _Surface()
        surface_entry = {
            "name": "outer",
            "seed_label": 0.16,
            "target_volume": 1.0,
            "boozer_surface": SimpleNamespace(
                surface=surface,
                res={"success": True, "iota": TEST_IOTA, "G": TEST_G0},
                save=lambda path: None,
            ),
        }
        objective_eval = {
            "total": 7.0,
            "grad": np.arange(3, dtype=float),
            "surface_weights": np.array([1.0]),
            "J_QS": 0.0,
            "dJ_QS": np.zeros(3),
            "J_Boozer": 0.0,
            "dJ_Boozer": np.zeros(3),
            "J_iota": 0.0,
            "dJ_iota": np.zeros(3),
            "J_curvature": 0.0,
            "dJ_curvature": np.zeros(3),
        }
        stack_status = {
            "success": True,
            "solve_success": [True],
            "self_intersections": [False],
            "volumes_ordered": True,
            "gap_ok": True,
            "nesting_ok": True,
            "adjacent_gaps": [],
            "outer_vessel_gap": None,
            "bad_nesting_phis": [],
        }
        accepted_surface_state = {
            "sdofs": [np.array([0.1])],
            "iota": [TEST_IOTA],
            "G": [TEST_G0],
        }
        hardware_snapshot = {
            "curve_curve_min_dist": 0.04,
            "curve_surface_min_dist": 0.03,
            "surface_vessel_min_dist": 0.0,
            "max_curvature": 41.0,
            "tf_current_A": -8.0e4,
            "success": False,
            "violations": ["coil_coil_min_dist=0.040000 < threshold=0.050000"],
            "search_hardware_status": {
                "success": False,
                "violations": ["coil_coil_min_dist=0.040000 < threshold=0.050000"],
            },
            "artifact_hardware_status": {
                "success": False,
                "violations": ["coil_coil_min_dist=0.040000 < threshold=0.050000"],
            },
        }

        module.surface_data = [surface_entry]
        module.outer_surface_data = surface_entry
        module.surface_iota_terms = [SimpleNamespace(J=lambda: TEST_IOTA)]
        module.JCurveLength = _ScalarObjective(0.44)
        module.JCurveCurve = _DistanceObjective(0.55, 0.04)
        module.JCurveSurface = _DistanceObjective(0.77, 0.03)
        module.JCurvature = _ScalarObjective(0.99)
        module.banana_curve = _Curve()
        module.banana_curves = [module.banana_curve]
        module.curvelength = _CurveLength()
        module.CurveLength = lambda curve: _CurveLength()
        module.bs = _BS()
        module.VV = object()
        module.CHECKPOINT_EVERY = 0
        module.TOPOLOGY_SCORER_EVERY = 0
        module.CONSTRAINT_METHOD = "penalty"
        # Preserved-timeout writes performed during callback stamp WOUT_CONVENTION
        # at producer time; pin the replay config so the helper can read a real
        # WOUT fixture (single_stage_banana_example.py:6657).
        module.PRESERVED_TIMEOUT_REPLAY_CONFIG = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_10x10.nc",
            plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
            stage2_bs_path="",
            stage2_results_path="",
            mpol=0,
            ntor=0,
            nphi=0,
            ntheta=0,
            constraint_weight=None,
            constraint_method=None,
            alm_formulation=None,
            max_iterations=None,
            target_volume=None,
            target_iota=None,
        )
        module.run_dict["surface_state"] = accepted_surface_state
        module.run_dict["it"] = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            module.OUT_DIR_ITER = tmpdir

            with (
                patch.object(
                    module,
                    "evaluate_search_objective",
                    return_value=objective_eval,
                ),
                patch.object(
                    module,
                    "snapshot_surface_states",
                    return_value=accepted_surface_state,
                ),
                patch.object(
                    module,
                    "evaluate_surface_stack",
                    return_value=stack_status,
                ),
                patch.object(
                    module,
                    "evaluate_single_stage_hardware_snapshot",
                    return_value=hardware_snapshot,
                ),
            ):
                module.callback(np.ones(3))

            self.assertFalse(module.run_dict["accepted_hardware_status"]["success"])
            log_text = (Path(tmpdir) / "log.txt").read_text()

        self.assertIn("Hardware Constraints OK", log_text)
        self.assertIn("Hardware Violations", log_text)
        self.assertIn("coil_coil_min_dist=0.040000 < threshold=0.050000", log_text)

    def test_alm_rejection_preserves_constraint_metadata_for_outer_updates(self):
        module = load_single_stage_example_module()
        module.CONSTRAINT_METHOD = "alm"
        module.MULTISURFACE_RAMP_ITERATIONS = 0
        module.INNER_SURFACE_INITIAL_WEIGHT = 1.0
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.TOPOLOGY_GATE_TMAX = 2.0
        module.TOPOLOGY_GATE_TOL = 1e-7
        module.TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.25
        module.JF = SimpleNamespace(x=np.zeros(2))
        module.surface_data = [{"boozer_surface": SimpleNamespace(surface=object())}]
        module.run_dict = {
            "x_prev": np.zeros(2),
            "lscount": 0,
            "accepted_iterations": 0,
            "surface_state": {"sdofs": [], "iota": [], "G": []},
            "accepted_x": np.zeros(2),
            "J": 7.0,
            "dJ": np.array([3.0, -1.0]),
            "search_eval": {
                "constraint_values": np.array([0.4, 0.1, 0.0]),
                "feasibility_values": np.array([0.4, 0.1, 0.0]),
                "dual_update_values": np.array([0.4, 0.1, 0.0]),
                "hard_signed_constraint_values": np.array([0.0, 0.02, -0.03]),
                "hard_violation_values": np.array([0.0, 0.02, 0.0]),
                "surrogate_signed_constraint_values": np.array([0.4, 0.1, 0.0]),
                "hard_dual_update_values": np.array([0.0, 0.02, -0.03]),
                "raw_hard_signed_constraint_values": np.array([0.0, 2.0, -3.0]),
                "raw_hard_violation_values": np.array([0.0, 2.0, 0.0]),
                "raw_surrogate_signed_constraint_values": np.array([40.0, 10.0, 0.0]),
                "raw_hard_dual_update_values": np.array([0.0, 2.0, -3.0]),
                "normalized_signed_constraint_values": np.array([0.4, 0.1, 0.0]),
                "normalized_feasibility_values": np.array([0.4, 0.1, 0.0]),
                "constraint_scales": np.array([100.0, 100.0, 100.0]),
                "constraint_blocks": ["geometry", "geometry", "geometry"],
                "constraint_scale_sources": [
                    "threshold:coil_coil_spacing",
                    "threshold:coil_surface_spacing",
                    "threshold:max_curvature",
                ],
                "base_grad": np.array([0.2, -0.3]),
                "max_violation": 0.4,
                "stationarity_norm": 2.5,
                "constraint_names": [
                    "coil_coil_spacing",
                    "coil_surface_spacing",
                    "max_curvature",
                ],
                "base_total": 5.0,
            },
        }

        with (
            patch.object(
                module,
                "solve_surface_stack_at_dofs",
                return_value={
                    "success": False,
                    "solve_success": [False],
                    "self_intersections": [False],
                    "volumes_ordered": True,
                    "gap_ok": True,
                    "nesting_ok": True,
                    "adjacent_gaps": [],
                    "outer_vessel_gap": None,
                    "bad_nesting_phis": [],
                },
            ),
            patch.object(module, "restore_surface_states") as restore_mock,
        ):
            evaluation = module.evaluate_search_step(np.ones(2))

        self.assertEqual(evaluation["total"], 14.0)
        np.testing.assert_array_equal(evaluation["grad"], np.array([3.0, -1.0]))
        np.testing.assert_array_equal(
            evaluation["constraint_values"],
            np.array([0.4, 0.1, 0.0]),
        )
        self.assertAlmostEqual(evaluation["max_violation"], 0.4)
        self.assertAlmostEqual(evaluation["stationarity_norm"], 2.5)
        self.assertEqual(
            evaluation["constraint_names"],
            ["coil_coil_spacing", "coil_surface_spacing", "max_curvature"],
        )
        np.testing.assert_array_equal(
            evaluation["hard_signed_constraint_values"],
            np.array([0.0, 0.02, -0.03]),
        )
        np.testing.assert_array_equal(
            evaluation["hard_violation_values"],
            np.array([0.0, 0.02, 0.0]),
        )
        np.testing.assert_array_equal(
            evaluation["surrogate_signed_constraint_values"],
            np.array([0.4, 0.1, 0.0]),
        )
        np.testing.assert_array_equal(
            evaluation["hard_dual_update_values"],
            np.array([0.0, 0.02, -0.03]),
        )
        np.testing.assert_array_equal(
            evaluation["raw_hard_signed_constraint_values"],
            np.array([0.0, 2.0, -3.0]),
        )
        np.testing.assert_array_equal(
            evaluation["constraint_scales"],
            np.array([100.0, 100.0, 100.0]),
        )
        self.assertEqual(
            evaluation["constraint_scale_sources"],
            [
                "threshold:coil_coil_spacing",
                "threshold:coil_surface_spacing",
                "threshold:max_curvature",
            ],
        )
        np.testing.assert_array_equal(evaluation["base_grad"], np.array([0.2, -0.3]))
        self.assertAlmostEqual(evaluation["base_total"], 5.0)
        restore_mock.assert_called_once()

    def test_evaluate_search_step_frontier_trust_excess_remains_search_penalty(self):
        module = self.load_module()

        class _Surface:
            def volume(self):
                return 0.10

        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_GOAL_CONFIG = SimpleNamespace(
            boozer_trust_threshold=1.0e-5,
            boozer_trust_penalty_scale=5.0e-5,
        )
        module.MULTISURFACE_RAMP_ITERATIONS = 0
        module.INNER_SURFACE_INITIAL_WEIGHT = 1.0
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.TOPOLOGY_GATE_TMAX = 2.0
        module.TOPOLOGY_GATE_TOL = 1.0e-3
        module.TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.5
        module.HARDWARE_SEARCH_MODE = "hard"
        module.HARDWARE_SEARCH_SOFT_ITERATIONS = 0
        module.CC_DIST = 0.05
        module.CS_DIST = 0.015
        module.CURVATURE_THRESHOLD = 40.0
        module.bs = object()
        module.JCurveCurve = object()
        module.JCurveSurface = object()
        module.banana_curve = object()
        module.JF = SimpleNamespace(x=np.zeros(2))
        seed_near_miss_diagnostic_globals(module)
        module.surface_iota_terms = [SimpleNamespace(J=lambda: 0.15)]
        module.surface_data = [{"boozer_surface": SimpleNamespace(surface=_Surface())}]
        module.run_dict = {
            "x_prev": np.zeros(2),
            "lscount": 0,
            "accepted_iterations": 0,
            "surface_state": {"sdofs": [], "iota": [], "G": []},
            "accepted_x": np.zeros(2),
            "J": 7.0,
            "dJ": np.array([3.0, -1.0]),
            "search_eval": {"total": 7.0},
            "invalid_state_rejects_total": 0,
            "topology_gate_rejects": 0,
            "hardware_rejects": 0,
            "surface_solve_rejects": 0,
            "frontier_trust_rejects": 0,
        }
        stack_status = {
            "success": True,
            "solve_success": [True],
            "self_intersections": [False],
            "volumes_ordered": True,
            "gap_ok": True,
            "nesting_ok": True,
            "adjacent_gaps": [],
            "outer_vessel_gap": None,
            "bad_nesting_phis": [],
        }
        objective_eval = {
            "total": 2.09,
            "grad": np.array([1.024, -2.012]),
            "J_Boozer": 2.5e-5,
            "dJ_Boozer": np.array([2.0e-6, -1.0e-6]),
            "frontier_trust_penalty": 0.09,
            "frontier_boozer_trust_excess_ratio": 0.3,
            "surface_weights": np.array([1.0]),
            **search_hardware_penalty_payload(),
        }

        with (
            patch.object(
                module,
                "solve_surface_stack_at_dofs",
                return_value=stack_status,
            ),
            patch.object(
                module,
                "evaluate_search_objective",
                return_value=objective_eval,
            ),
            patch.object(
                module,
                "evaluate_search_topology_gate",
                return_value={"enabled": False, "success": True},
            ),
            patch.object(
                module,
                "evaluate_surface_stack",
                return_value=stack_status,
            ),
            patch.object(
                module,
                "evaluate_single_stage_hardware_snapshot",
                return_value=topology_hardware_snapshot(),
            ),
            patch.object(
                module,
                "compute_surface_field_metrics",
                return_value=(0.0, 0.0),
            ),
            patch.object(
                module,
                "maybe_write_best_hardware_near_miss_trial_artifacts",
                return_value=False,
            ),
        ):
            evaluation = module.evaluate_search_step(np.ones(2))

        self.assertAlmostEqual(evaluation["total"], 2.09)
        np.testing.assert_allclose(evaluation["grad"], [1.024, -2.012])
        self.assertEqual(module.run_dict["invalid_state_rejects_total"], 0)
        self.assertEqual(module.run_dict["frontier_trust_rejects"], 1)
        self.assertTrue(module.run_dict["topology_gate_status"]["success"])

    def test_evaluate_search_step_frontier_topology_reject_becomes_penalty(self):
        module = self.load_module()

        class _Surface:
            def volume(self):
                return 0.10

        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_GOAL_CONFIG = SimpleNamespace(
            boozer_trust_threshold=1.0e-5,
            boozer_trust_penalty_scale=5.0e-5,
        )
        module.MULTISURFACE_RAMP_ITERATIONS = 0
        module.INNER_SURFACE_INITIAL_WEIGHT = 1.0
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.TOPOLOGY_GATE_TMAX = 2.0
        module.TOPOLOGY_GATE_TOL = 1.0e-3
        module.TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.5
        module.TOPOLOGY_GATE_PENALTY_SCALE = 4.0
        module.HARDWARE_SEARCH_MODE = "hard"
        module.HARDWARE_SEARCH_SOFT_ITERATIONS = 0
        module.CC_DIST = 0.05
        module.CS_DIST = 0.015
        module.CURVATURE_THRESHOLD = 40.0
        module.bs = object()
        module.JCurveCurve = object()
        module.JCurveSurface = object()
        module.banana_curve = object()
        module.JF = SimpleNamespace(x=np.zeros(2))
        seed_near_miss_diagnostic_globals(module)
        module.surface_iota_terms = [SimpleNamespace(J=lambda: 0.15)]
        module.surface_data = [{"boozer_surface": SimpleNamespace(surface=_Surface())}]
        module.run_dict = {
            "x_prev": np.zeros(2),
            "lscount": 0,
            "accepted_iterations": 0,
            "surface_state": {"sdofs": [], "iota": [], "G": []},
            "accepted_x": np.zeros(2),
            "J": 7.0,
            "dJ": np.array([3.0, -1.0]),
            "search_eval": {"total": 7.0},
            "invalid_state_rejects_total": 0,
            "topology_gate_rejects": 0,
            "hardware_rejects": 0,
            "surface_solve_rejects": 0,
            "frontier_trust_rejects": 0,
        }
        stack_status = {
            "success": True,
            "solve_success": [True],
            "self_intersections": [False],
            "volumes_ordered": True,
            "gap_ok": True,
            "nesting_ok": True,
            "adjacent_gaps": [],
            "outer_vessel_gap": None,
            "bad_nesting_phis": [],
        }
        objective_eval = {
            "total": 2.0,
            "grad": np.array([1.0, -2.0]),
            "J_Boozer": 5.0e-6,
            "dJ_Boozer": np.array([0.0, 0.0]),
            "surface_weights": np.array([1.0]),
            **search_hardware_penalty_payload(),
        }

        with (
            patch.object(
                module,
                "solve_surface_stack_at_dofs",
                return_value=stack_status,
            ),
            patch.object(
                module,
                "evaluate_search_objective",
                return_value=objective_eval,
            ),
            patch.object(
                module,
                "evaluate_search_topology_gate",
                return_value={
                    "enabled": True,
                    "success": False,
                    "survived_lines": 1,
                    "nfieldlines": 4,
                    "survival_fraction": 0.5,
                    "survival_threshold": 0.75,
                    "first_exit_time": None,
                    "first_exit_angle": None,
                    "first_exit_reason": None,
                },
            ),
            patch.object(
                module,
                "evaluate_single_stage_hardware_snapshot",
                side_effect=AssertionError(
                    "exact hardware snapshot should not be evaluated"
                ),
            ),
        ):
            evaluation = module.evaluate_search_step(np.ones(2))

        self.assertAlmostEqual(evaluation["total"], 9.0)
        np.testing.assert_allclose(evaluation["grad"], [1.0, -2.0])
        self.assertAlmostEqual(
            module.run_dict["last_successful_eval"]["frontier_topology_penalty"], 7.0
        )
        self.assertAlmostEqual(
            module.run_dict["last_successful_eval"]["frontier_contract_penalty"], 7.0
        )
        self.assertEqual(module.run_dict["invalid_state_rejects_total"], 0)
        self.assertEqual(module.run_dict["topology_gate_rejects"], 0)
        self.assertFalse(module.run_dict["topology_gate_status"]["success"])

    def test_evaluate_search_step_frontier_hardware_reject_becomes_penalty(self):
        module = self.load_module()

        class _Surface:
            def volume(self):
                return 0.10

        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_GOAL_CONFIG = SimpleNamespace(
            boozer_trust_threshold=1.0e-5,
            boozer_trust_penalty_scale=5.0e-5,
        )
        module.MULTISURFACE_RAMP_ITERATIONS = 0
        module.INNER_SURFACE_INITIAL_WEIGHT = 1.0
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.TOPOLOGY_GATE_TMAX = 2.0
        module.TOPOLOGY_GATE_TOL = 1.0e-3
        module.TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.5
        module.HARDWARE_SEARCH_MODE = "hard"
        module.HARDWARE_SEARCH_SOFT_ITERATIONS = 0
        module.HARDWARE_SEARCH_PENALTY_SCALE = 4.0
        module.CC_DIST = 0.05
        module.CS_DIST = 0.015
        module.CURVATURE_THRESHOLD = 40.0
        module.bs = object()
        module.JCurveCurve = object()
        module.JCurveSurface = object()
        module.banana_curve = object()
        module.JF = SimpleNamespace(x=np.zeros(2))
        seed_near_miss_diagnostic_globals(module)
        module.surface_iota_terms = [SimpleNamespace(J=lambda: 0.15)]
        module.surface_data = [{"boozer_surface": SimpleNamespace(surface=_Surface())}]
        module.run_dict = {
            "x_prev": np.zeros(2),
            "lscount": 0,
            "accepted_iterations": 0,
            "surface_state": {"sdofs": [], "iota": [], "G": []},
            "accepted_x": np.zeros(2),
            "J": 7.0,
            "dJ": np.array([3.0, -1.0]),
            "search_eval": {"total": 7.0},
            "invalid_state_rejects_total": 0,
            "topology_gate_rejects": 0,
            "hardware_rejects": 0,
            "surface_solve_rejects": 0,
            "frontier_trust_rejects": 0,
        }
        stack_status = {
            "success": True,
            "solve_success": [True],
            "self_intersections": [False],
            "volumes_ordered": True,
            "gap_ok": True,
            "nesting_ok": True,
            "adjacent_gaps": [],
            "outer_vessel_gap": None,
            "bad_nesting_phis": [],
        }
        objective_eval = {
            "total": 2.0,
            "grad": np.array([1.0, -2.0]),
            "J_Boozer": 5.0e-6,
            "dJ_Boozer": np.array([0.0, 0.0]),
            "surface_weights": np.array([1.0]),
            **search_hardware_penalty_payload((0.25, 0.0, 0.0)),
        }

        with (
            patch.object(
                module,
                "solve_surface_stack_at_dofs",
                return_value=stack_status,
            ),
            patch.object(
                module,
                "evaluate_search_objective",
                return_value=objective_eval,
            ),
            patch.object(
                module,
                "evaluate_search_topology_gate",
                return_value={"enabled": False, "success": True},
            ),
            patch.object(
                module,
                "evaluate_surface_stack",
                return_value=stack_status,
            ),
            patch.object(
                module,
                "evaluate_single_stage_hardware_snapshot",
                return_value=topology_hardware_snapshot(),
            ),
            patch.object(
                module,
                "compute_surface_field_metrics",
                return_value=(0.0, 0.0),
            ),
            patch.object(
                module,
                "maybe_write_best_hardware_near_miss_trial_artifacts",
                return_value=False,
            ),
        ):
            evaluation = module.evaluate_search_step(np.ones(2))

        self.assertAlmostEqual(evaluation["total"], 9.0)
        np.testing.assert_allclose(evaluation["grad"], [1.0, -2.0])
        self.assertAlmostEqual(
            module.run_dict["last_successful_eval"]["frontier_hardware_penalty"], 7.0
        )
        self.assertAlmostEqual(
            module.run_dict["last_successful_eval"]["frontier_contract_penalty"], 7.0
        )
        self.assertEqual(module.run_dict["invalid_state_rejects_total"], 0)
        self.assertEqual(module.run_dict["hardware_rejects"], 0)
        self.assertFalse(module.run_dict["trial_hardware_status"]["success"])

    def test_evaluate_search_step_frontier_hardware_penalty_sees_live_width_self_intersect(
        self,
    ):
        module = self.load_module()

        class _Surface:
            def volume(self):
                return 0.10

        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_GOAL_CONFIG = SimpleNamespace(
            boozer_trust_threshold=1.0e-5,
            boozer_trust_penalty_scale=5.0e-5,
        )
        module.MULTISURFACE_RAMP_ITERATIONS = 0
        module.INNER_SURFACE_INITIAL_WEIGHT = 1.0
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.TOPOLOGY_GATE_TMAX = 2.0
        module.TOPOLOGY_GATE_TOL = 1.0e-3
        module.TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.5
        module.HARDWARE_SEARCH_MODE = "hard"
        module.HARDWARE_SEARCH_SOFT_ITERATIONS = 0
        module.HARDWARE_SEARCH_PENALTY_SCALE = 4.0
        module.CC_DIST = 0.05
        module.CS_DIST = 0.015
        module.CURVATURE_THRESHOLD = 40.0
        module.bs = object()
        module.JCurveCurve = object()
        module.JCurveSurface = object()
        module.banana_curve = object()
        module.JF = SimpleNamespace(x=np.zeros(2))
        module.JCoilWidth = SimpleNamespace(J=lambda: 0.02)
        module.JCurveSelfIntersect = SimpleNamespace(J=lambda: 0.25)
        module.JPoloidalExtent = None
        module.surface_iota_terms = [SimpleNamespace(J=lambda: 0.15)]
        seed_near_miss_diagnostic_globals(module)
        module.surface_data = [{"boozer_surface": SimpleNamespace(surface=_Surface())}]
        module.run_dict = {
            "x_prev": np.zeros(2),
            "lscount": 0,
            "accepted_iterations": 0,
            "surface_state": {"sdofs": [], "iota": [], "G": []},
            "accepted_x": np.zeros(2),
            "J": 7.0,
            "dJ": np.array([3.0, -1.0]),
            "search_eval": {"total": 7.0},
            "invalid_state_rejects_total": 0,
            "topology_gate_rejects": 0,
            "hardware_rejects": 0,
            "surface_solve_rejects": 0,
            "frontier_trust_rejects": 0,
        }
        stack_status = {
            "success": True,
            "solve_success": [True],
            "self_intersections": [False],
            "volumes_ordered": True,
            "gap_ok": True,
            "nesting_ok": True,
            "adjacent_gaps": [],
            "outer_vessel_gap": None,
            "bad_nesting_phis": [],
        }
        objective_eval = {
            "total": 2.0,
            "grad": np.array([1.0, -2.0]),
            "J_Boozer": 5.0e-6,
            "dJ_Boozer": np.array([0.0, 0.0]),
            "surface_weights": np.array([1.0]),
            **search_hardware_penalty_payload((0.0, 0.0, 0.0)),
        }

        with (
            patch.object(
                module,
                "solve_surface_stack_at_dofs",
                return_value=stack_status,
            ),
            patch.object(
                module,
                "evaluate_search_objective",
                return_value=objective_eval,
            ),
            patch.object(
                module,
                "evaluate_search_topology_gate",
                return_value={"enabled": False, "success": True},
            ),
            patch.object(
                module,
                "evaluate_surface_stack",
                return_value=stack_status,
            ),
            patch.object(
                module,
                "evaluate_single_stage_hardware_snapshot",
                return_value=topology_hardware_snapshot(),
            ),
            patch.object(
                module,
                "compute_surface_field_metrics",
                return_value=(0.0, 0.0),
            ),
            patch.object(
                module,
                "maybe_write_best_hardware_near_miss_trial_artifacts",
                return_value=False,
            ),
        ):
            evaluation = module.evaluate_search_step(np.ones(2))

        hardware_status = module.run_dict["trial_hardware_status"]
        ratios = module.run_dict["last_successful_eval"][
            "frontier_hardware_violation_ratios"
        ]
        self.assertFalse(hardware_status["success"])
        self.assertIn("width_min", hardware_status["constraints"])
        self.assertIn("self_intersect", hardware_status["constraints"])
        self.assertAlmostEqual(ratios["width_min"], 0.8)
        self.assertAlmostEqual(ratios["self_intersect"], 0.25)
        self.assertAlmostEqual(
            module.run_dict["last_successful_eval"]["frontier_hardware_penalty"],
            22.4,
        )
        self.assertAlmostEqual(evaluation["total"], 24.4)

    def test_evaluate_search_step_repair_phase1_keeps_valid_hardware_bad_candidate_live(
        self,
    ):
        module = self.load_module()

        class _Surface:
            def volume(self):
                return 0.10

        module.SINGLE_STAGE_GOAL_MODE = "target"
        module.MULTISURFACE_RAMP_ITERATIONS = 0
        module.INNER_SURFACE_INITIAL_WEIGHT = 1.0
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.TOPOLOGY_GATE_TMAX = 2.0
        module.TOPOLOGY_GATE_TOL = 1.0e-3
        module.TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.5
        module.HARDWARE_SEARCH_MODE = "hard"
        module.HARDWARE_SEARCH_SOFT_ITERATIONS = 0
        module.CC_DIST = 0.05
        module.CS_DIST = 0.015
        module.CURVATURE_THRESHOLD = 40.0
        module.bs = object()
        module.JCurveCurve = object()
        module.JCurveSurface = object()
        module.banana_curve = object()
        module.JF = SimpleNamespace(x=np.zeros(2))
        seed_near_miss_diagnostic_globals(module)
        module.surface_iota_terms = [SimpleNamespace(J=lambda: 0.15)]
        module.surface_data = [{"boozer_surface": SimpleNamespace(surface=_Surface())}]
        module.run_dict = {
            "x_prev": np.zeros(2),
            "lscount": 0,
            "accepted_iterations": 0,
            "surface_state": {"sdofs": [], "iota": [], "G": []},
            "accepted_x": np.zeros(2),
            "J": 7.0,
            "dJ": np.array([3.0, -1.0]),
            "search_eval": {"total": 7.0},
            "invalid_state_rejects_total": 0,
            "topology_gate_rejects": 0,
            "hardware_rejects": 0,
            "surface_solve_rejects": 0,
            "frontier_trust_rejects": 0,
            "phase1_repair_mode_active": True,
        }
        stack_status = {
            "success": True,
            "solve_success": [True],
            "self_intersections": [False],
            "volumes_ordered": True,
            "gap_ok": True,
            "nesting_ok": True,
            "adjacent_gaps": [],
            "outer_vessel_gap": None,
            "bad_nesting_phis": [],
        }
        objective_eval = {
            "total": 2.0,
            "grad": np.array([1.0, -2.0]),
            "J_cc": 1.5,
            "dJ_cc": np.array([0.5, -0.5]),
            "J_cs": 0.5,
            "dJ_cs": np.array([0.25, -0.25]),
            "J_curvature": 0.25,
            "dJ_curvature": np.array([0.1, -0.1]),
            "surface_weights": np.array([1.0]),
            **search_hardware_penalty_payload((0.1, 0.0, 0.0)),
        }

        with (
            patch.object(
                module,
                "solve_surface_stack_at_dofs",
                return_value=stack_status,
            ),
            patch.object(
                module,
                "evaluate_search_objective",
                return_value=objective_eval,
            ),
            patch.object(
                module,
                "evaluate_search_topology_gate",
                return_value={"enabled": False, "success": True},
            ),
            patch.object(
                module,
                "evaluate_surface_stack",
                return_value=stack_status,
            ),
            patch.object(
                module,
                "evaluate_single_stage_hardware_snapshot",
                return_value=topology_hardware_snapshot(),
            ),
            patch.object(
                module,
                "compute_surface_field_metrics",
                return_value=(0.0, 0.0),
            ),
            patch.object(
                module,
                "maybe_write_best_hardware_near_miss_trial_artifacts",
                return_value=False,
            ),
        ):
            evaluation = module.evaluate_search_step(np.ones(2))

        self.assertAlmostEqual(evaluation["total"], 2.0)
        np.testing.assert_allclose(evaluation["grad"], [1.0, -2.0])
        self.assertEqual(module.run_dict["invalid_state_rejects_total"], 0)
        self.assertEqual(module.run_dict["hardware_rejects"], 0)
        self.assertFalse(module.run_dict["trial_hardware_status"]["success"])
        self.assertEqual(module.run_dict["last_successful_eval"], objective_eval)

    def test_build_scaled_outer_problem_scales_coordinates_gradients_and_callback(self):
        module = self.load_module()
        seen = {"fun": [], "callback": []}

        def base_fun(x):
            seen["fun"].append(np.asarray(x, dtype=float).copy())
            return 7.5, np.array([3.0, -4.0])

        def base_callback(x):
            seen["callback"].append(np.asarray(x, dtype=float).copy())

        scaled_fun, scaled_callback = module.build_scaled_outer_problem(
            base_fun,
            base_callback,
            np.array([10.0, 20.0]),
            0.1,
        )

        J, dJ = scaled_fun(np.array([1.0, -2.0]))
        self.assertAlmostEqual(J, 7.5)
        np.testing.assert_allclose(dJ, [0.3, -0.4])
        np.testing.assert_allclose(seen["fun"][0], [10.1, 19.8])

        scaled_callback(np.array([1.0, -2.0]))
        np.testing.assert_allclose(seen["callback"][0], [10.1, 19.8])

    def test_build_scipy_bounds_returns_none_when_unbounded(self):
        module = self.load_module()

        bounds = module.build_scipy_bounds(
            np.array([-np.inf, -np.inf]),
            np.array([np.inf, np.inf]),
        )

        self.assertIsNone(bounds)

    def test_build_scaled_outer_bounds_transforms_to_scaled_coordinates(self):
        module = self.load_module()

        bounds = module.build_scaled_outer_bounds(
            np.array([10.0, 20.0]),
            0.1,
            np.array([9.0, -np.inf]),
            np.array([10.5, 25.0]),
        )

        self.assertEqual(bounds, [(-10.0, 5.0), (-np.inf, 50.0)])

    def test_build_local_relative_bounds_clips_to_anchor_box_and_global_bounds(self):
        module = self.load_module()

        bounds = module.build_local_relative_bounds(
            np.array([10.0, -2.0]),
            0.1,
            np.array([9.5, -10.0]),
            np.array([12.0, -1.5]),
        )

        self.assertEqual(bounds, [(9.5, 11.0), (-2.2, -1.8)])

    def test_build_scaled_local_outer_bounds_transforms_local_box(self):
        module = self.load_module()

        bounds = module.build_scaled_local_outer_bounds(
            np.array([10.0, 20.0]),
            0.5,
            np.array([8.0, 18.0]),
            np.array([15.0, 30.0]),
            0.1,
        )

        self.assertEqual(bounds, [(-2.0, 2.0), (-4.0, 4.0)])

    def test_resolve_initial_step_phase_maxiter(self):
        module = self.load_module()

        self.assertEqual(module.resolve_initial_step_phase_maxiter(40, 1.0, 10), 0)
        self.assertEqual(module.resolve_initial_step_phase_maxiter(40, 0.5, 0), 0)
        self.assertEqual(module.resolve_initial_step_phase_maxiter(40, 0.5, 10), 10)
        self.assertEqual(module.resolve_initial_step_phase_maxiter(5, 0.5, 10), 5)

    def test_penalty_feasible_start_local_preservation_enabled(self):
        module = self.load_module()
        run_dict = {
            "accepted_iterations": 0,
            "accepted_hardware_status": {"success": True},
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
        }

        self.assertTrue(
            module.penalty_feasible_start_local_preservation_enabled(
                run_dict,
                constraint_method="penalty",
                num_surfaces=1,
                basin_hops=0,
                init_only=False,
            )
        )
        self.assertFalse(
            module.penalty_feasible_start_local_preservation_enabled(
                run_dict,
                constraint_method="penalty",
                num_surfaces=2,
                basin_hops=0,
                init_only=False,
            )
        )

    def test_resolve_single_stage_seed_regime_auto_routes_by_incumbent_state(self):
        module = self.load_module()
        good_run_dict = {
            "accepted_hardware_status": {"success": True},
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
        }
        bridge_run_dict = {
            "accepted_hardware_status": {"success": False},
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
        }
        bad_run_dict = {
            "accepted_hardware_status": {"success": False},
            "surface_status": {"success": False},
            "intersecting": True,
            "search_eval": {"total": 1.0},
        }

        self.assertEqual(
            module.resolve_single_stage_seed_regime(
                "auto",
                good_run_dict,
                constraint_method="penalty",
                num_surfaces=1,
                basin_hops=0,
                init_only=False,
            ),
            "preserve_first",
        )
        self.assertEqual(
            module.resolve_single_stage_seed_regime(
                "auto",
                bridge_run_dict,
                constraint_method="penalty",
                num_surfaces=1,
                basin_hops=0,
                init_only=False,
            ),
            "bridge_only",
        )
        self.assertEqual(
            module.resolve_single_stage_seed_regime(
                "auto",
                bad_run_dict,
                constraint_method="penalty",
                num_surfaces=1,
                basin_hops=0,
                init_only=False,
            ),
            "repair_first",
        )
        self.assertEqual(
            module.resolve_single_stage_seed_regime(
                "bridge_only",
                good_run_dict,
                constraint_method="alm",
                num_surfaces=1,
                basin_hops=0,
                init_only=False,
            ),
            "global_search",
        )

    def test_resolve_penalty_phase1_settings_auto_enables_local_preservation(self):
        module = self.load_module()

        settings = module.resolve_penalty_phase1_settings(
            40,
            1.0,
            0,
            enable_local_preservation=True,
        )

        self.assertTrue(settings["use_phase1"])
        self.assertTrue(settings["auto_enabled"])
        self.assertTrue(settings["use_local_bounds"])
        self.assertEqual(
            settings["phase1_maxiter"],
            min(40, module._PENALTY_FEASIBLE_START_LOCAL_MAXITER),
        )
        self.assertEqual(settings["phase1_scale"], 1.0)
        self.assertEqual(
            settings["local_relative_radius"],
            module._PENALTY_FEASIBLE_START_LOCAL_RELATIVE_RADIUS,
        )

    def test_resolve_penalty_phase1_settings_frontier_auto_contracts_phase1(self):
        module = self.load_module()

        settings = module.resolve_penalty_phase1_settings(
            40,
            1.0,
            0,
            enable_local_preservation=True,
            is_frontier_mode=True,
        )

        self.assertTrue(settings["use_phase1"])
        self.assertTrue(settings["auto_enabled"])
        self.assertTrue(settings["use_local_bounds"])
        self.assertEqual(
            settings["phase1_maxiter"],
            min(40, module._PENALTY_FEASIBLE_START_LOCAL_MAXITER),
        )
        self.assertEqual(
            settings["phase1_scale"],
            module._FRONTIER_FEASIBLE_START_PHASE1_SCALE,
        )
        self.assertEqual(
            settings["local_relative_radius"],
            module._FRONTIER_FEASIBLE_START_LOCAL_RELATIVE_RADIUS,
        )

    def test_resolve_penalty_phase1_settings_frontier_does_not_auto_contract_when_local_preservation_disabled(
        self,
    ):
        module = self.load_module()

        settings = module.resolve_penalty_phase1_settings(
            40,
            1.0,
            0,
            enable_local_preservation=False,
            is_frontier_mode=True,
        )

        self.assertFalse(settings["use_phase1"])
        self.assertFalse(settings["auto_enabled"])
        self.assertFalse(settings["use_local_bounds"])
        self.assertEqual(settings["phase1_scale"], 1.0)
        self.assertIsNone(settings["local_relative_radius"])

    def test_resolve_penalty_phase1_settings_frontier_respects_explicit_initial_step_scale(
        self,
    ):
        module = self.load_module()

        settings = module.resolve_penalty_phase1_settings(
            40,
            0.5,
            5,
            enable_local_preservation=True,
            is_frontier_mode=True,
        )

        self.assertTrue(settings["use_phase1"])
        self.assertFalse(settings["auto_enabled"])
        self.assertTrue(settings["use_local_bounds"])
        self.assertEqual(settings["phase1_maxiter"], 5)
        self.assertEqual(settings["phase1_scale"], 0.5)
        self.assertEqual(
            settings["local_relative_radius"],
            module._PENALTY_FEASIBLE_START_LOCAL_RELATIVE_RADIUS,
        )

    def test_resolve_penalty_phase1_settings_repair_first_uses_extra_local_attempt(
        self,
    ):
        module = self.load_module()

        settings = module.resolve_penalty_phase1_settings(
            40,
            1.0,
            0,
            enable_local_preservation=True,
            seed_regime="repair_first",
        )

        self.assertTrue(settings["use_phase1"])
        self.assertTrue(settings["use_local_bounds"])
        self.assertEqual(
            settings["local_max_attempts"],
            module._PENALTY_FEASIBLE_START_LOCAL_MAX_ATTEMPTS + 1,
        )

    def test_run_penalty_phase1_preserves_feasible_start_when_no_safe_step_exists(self):
        module = self.load_module()
        run_dict = {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": {"success": True},
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
        }
        restore_calls = []

        def fake_restore():
            restore_calls.append(True)

        def fake_minimize(*args, **kwargs):
            return SimpleNamespace(
                nit=2, success=False, message="ABNORMAL_TERMINATION", status=2
            )

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=True,
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=lambda x: None,
            normalize_message_fn=lambda *args, **kwargs: "phase1_reject",
            restore_accepted_state_fn=fake_restore,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        self.assertTrue(result["used_phase1"])
        self.assertFalse(result["continue_search"])
        self.assertTrue(result["local_preservation_used"])
        self.assertTrue(result["local_preservation_preserved_start"])
        self.assertGreaterEqual(result["local_preservation_attempts"], 1)
        self.assertEqual(result["phase1_outcome"], "preserved_start_no_safe_step")
        self.assertIsNone(result["phase1_first_accepted_step_rms"])
        self.assertIsNone(result["phase1_max_accepted_step_rms"])
        self.assertFalse(result["phase1_anchor_restore_used"])
        self.assertEqual(result["phase1_unsafe_accept_rollbacks"], 0)
        self.assertEqual(result["phase1_invalid_reject_attempts"], 0)
        self.assertFalse(result["phase1_recovery_used"])
        self.assertEqual(result["next_dofs"].tolist(), [1.0, -1.0])
        self.assertGreaterEqual(len(restore_calls), 1)

    def test_run_penalty_phase1_continues_after_local_acceptance(self):
        module = self.load_module()
        run_dict = {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": {"success": True},
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
        }

        def fake_callback(x):
            run_dict["accepted_iterations"] += 1
            run_dict["accepted_x"] = np.asarray(x, dtype=float).copy()
            run_dict["accepted_hardware_status"] = {"success": True}

        def fake_minimize(fun, x0, **kwargs):
            kwargs["callback"](np.array([1.01, -0.99]))
            return SimpleNamespace(nit=1, success=True, message="CONVERGENCE", status=0)

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=True,
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=fake_callback,
            normalize_message_fn=lambda *args, **kwargs: "phase1_ok",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        self.assertTrue(result["continue_search"])
        self.assertFalse(result["local_preservation_preserved_start"])
        self.assertEqual(result["phase1_outcome"], "safe_local_accept")
        self.assertFalse(result["phase1_anchor_restore_used"])
        self.assertEqual(result["phase1_unsafe_accept_rollbacks"], 0)
        self.assertEqual(result["phase1_invalid_reject_attempts"], 0)
        self.assertFalse(result["phase1_recovery_used"])
        np.testing.assert_allclose(result["next_dofs"], [1.01, -0.99])
        expected_step_rms = module.basin_normalized_step_rms(
            np.array([1.0, -1.0]),
            np.array([1.01, -0.99]),
        )
        self.assertAlmostEqual(
            result["phase1_first_accepted_step_rms"],
            expected_step_rms,
        )
        self.assertAlmostEqual(
            result["phase1_max_accepted_step_rms"],
            expected_step_rms,
        )
        self.assertLessEqual(
            result["local_preservation_radius"],
            module._PENALTY_FEASIBLE_START_SAFE_STEP_RMS_LIMIT
            * module._PENALTY_FEASIBLE_START_PHASE2_RADIUS_SCALE,
        )

    def test_run_penalty_phase1_zero_move_accept_does_not_graduate(self):
        module = self.load_module()
        run_dict = {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": {"success": True},
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
        }

        def fake_callback(x):
            run_dict["accepted_iterations"] += 1
            run_dict["accepted_x"] = np.asarray(x, dtype=float).copy()

        def fake_minimize(fun, x0, **kwargs):
            kwargs["callback"](np.array([1.0, -1.0]))
            return SimpleNamespace(nit=1, success=True, message="CONVERGENCE", status=0)

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=True,
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=fake_callback,
            normalize_message_fn=lambda *args, **kwargs: "phase1_ok",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(
                module,
                phase1_config=module.build_phase1_config(local_max_attempts=1),
            ),
        )

        self.assertFalse(result["continue_search"])
        self.assertTrue(result["local_preservation_preserved_start"])
        self.assertEqual(result["phase1_outcome"], "preserved_start_no_safe_step")
        self.assertTrue(result["phase1_anchor_restore_used"])
        self.assertEqual(result["phase1_unsafe_accept_rollbacks"], 1)
        self.assertEqual(result["phase1_invalid_reject_attempts"], 0)
        self.assertTrue(result["phase1_recovery_used"])
        self.assertEqual(result["phase1_first_accepted_step_rms"], 0.0)
        self.assertEqual(result["phase1_max_accepted_step_rms"], 0.0)
        self.assertIn("unsafe_local_accept", result["phase1_termination_message"])

    def test_run_penalty_phase1_rolls_back_unsafe_accept_and_uses_reject_shrink(self):
        module = self.load_module()
        run_dict = {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": {"success": True},
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
        }
        seen_bounds = []
        attempts = {"count": 0}
        refresh_calls = []

        def fake_callback(x):
            run_dict["accepted_iterations"] += 1
            run_dict["accepted_x"] = np.asarray(x, dtype=float).copy()
            if attempts["count"] == 1:
                run_dict["accepted_hardware_status"] = {"success": False}
                run_dict["invalid_state_rejects_total"] += 1
            else:
                run_dict["accepted_hardware_status"] = {"success": True}

        def fake_minimize(fun, x0, **kwargs):
            attempts["count"] += 1
            seen_bounds.append(kwargs["bounds"])
            if attempts["count"] == 1:
                kwargs["callback"](np.array([1.04, -0.96]))
            else:
                kwargs["callback"](np.array([1.01, -0.99]))
            return SimpleNamespace(nit=1, success=True, message="CONVERGENCE", status=0)

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=True,
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=fake_callback,
            normalize_message_fn=lambda *args, **kwargs: "phase1_ok",
            restore_accepted_state_fn=lambda: None,
            refresh_preserved_timeout_artifacts_fn=lambda: refresh_calls.append(
                "refresh"
            ),
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        self.assertEqual(attempts["count"], 2)
        self.assertEqual(refresh_calls, ["refresh"])
        self.assertTrue(result["continue_search"])
        self.assertFalse(result["local_preservation_preserved_start"])
        self.assertEqual(result["phase1_outcome"], "safe_local_accept_after_recovery")
        self.assertTrue(result["phase1_anchor_restore_used"])
        self.assertEqual(result["phase1_unsafe_accept_rollbacks"], 1)
        self.assertEqual(result["phase1_invalid_reject_attempts"], 1)
        self.assertTrue(result["phase1_recovery_used"])
        np.testing.assert_allclose(result["next_dofs"], [1.01, -0.99])
        first_step_rms = module.basin_normalized_step_rms(
            np.array([1.0, -1.0]),
            np.array([1.04, -0.96]),
        )
        second_step_rms = module.basin_normalized_step_rms(
            np.array([1.0, -1.0]),
            np.array([1.01, -0.99]),
        )
        self.assertAlmostEqual(
            result["phase1_first_accepted_step_rms"],
            first_step_rms,
        )
        self.assertAlmostEqual(
            result["phase1_max_accepted_step_rms"],
            max(first_step_rms, second_step_rms),
        )
        self.assertEqual(seen_bounds[0], [(0.95, 1.05), (-1.05, -0.95)])
        self.assertEqual(
            seen_bounds[1],
            [
                (
                    1.0
                    - module._PENALTY_FEASIBLE_START_REJECT_RADIUS_SHRINK
                    * module._PENALTY_FEASIBLE_START_LOCAL_RELATIVE_RADIUS,
                    1.0
                    + module._PENALTY_FEASIBLE_START_REJECT_RADIUS_SHRINK
                    * module._PENALTY_FEASIBLE_START_LOCAL_RELATIVE_RADIUS,
                ),
                (
                    -1.0
                    - module._PENALTY_FEASIBLE_START_REJECT_RADIUS_SHRINK
                    * module._PENALTY_FEASIBLE_START_LOCAL_RELATIVE_RADIUS,
                    -1.0
                    + module._PENALTY_FEASIBLE_START_REJECT_RADIUS_SHRINK
                    * module._PENALTY_FEASIBLE_START_LOCAL_RELATIVE_RADIUS,
                ),
            ],
        )

    def test_run_penalty_phase1_tracks_first_and_max_accepted_step_across_callbacks(
        self,
    ):
        module = self.load_module()
        run_dict = {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": {"success": True},
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
        }

        def fake_callback(x):
            run_dict["accepted_iterations"] += 1
            run_dict["accepted_x"] = np.asarray(x, dtype=float).copy()
            run_dict["accepted_hardware_status"] = {"success": True}

        def fake_minimize(fun, x0, **kwargs):
            kwargs["callback"](np.array([1.015, -0.985]))
            kwargs["callback"](np.array([1.01, -0.99]))
            return SimpleNamespace(nit=2, success=True, message="CONVERGENCE", status=0)

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=True,
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=fake_callback,
            normalize_message_fn=lambda *args, **kwargs: "phase1_ok",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        self.assertEqual(result["phase1_outcome"], "safe_local_accept")
        self.assertAlmostEqual(result["phase1_first_accepted_step_rms"], 0.015)
        self.assertAlmostEqual(result["phase1_max_accepted_step_rms"], 0.015)
        np.testing.assert_allclose(result["next_dofs"], [1.01, -0.99])

    def test_run_penalty_phase1_reports_active_bounds_in_real_coordinates(self):
        module = self.load_module()
        run_dict = {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": {"success": True},
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
            "active_optimizer_bounds": [(-5.0, 5.0), (-5.0, 5.0)],
        }
        captured = {}

        def fake_callback(x):
            captured["callback_x"] = np.asarray(x, dtype=float).copy()
            captured["active_bounds"] = list(run_dict["active_optimizer_bounds"])
            run_dict["accepted_iterations"] += 1
            run_dict["accepted_x"] = np.asarray(x, dtype=float).copy()
            run_dict["accepted_hardware_status"] = {"success": True}

        def fake_minimize(fun, x0, **kwargs):
            captured["solver_bounds"] = kwargs["bounds"]
            kwargs["callback"](np.array([0.1, -0.1]))
            return SimpleNamespace(nit=1, success=True, message="CONVERGENCE", status=0)

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=0.5,
            initial_step_maxiter=1,
            enable_local_preservation=True,
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=fake_callback,
            normalize_message_fn=lambda *args, **kwargs: "phase1_ok",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        self.assertEqual(len(captured["solver_bounds"]), 2)
        np.testing.assert_allclose(
            np.asarray(captured["solver_bounds"], dtype=float),
            np.array([[-0.1, 0.1], [-0.1, 0.1]], dtype=float),
        )
        np.testing.assert_allclose(
            np.asarray(captured["active_bounds"], dtype=float),
            np.array([[0.95, 1.05], [-1.05, -0.95]], dtype=float),
        )
        np.testing.assert_allclose(captured["callback_x"], [1.05, -1.05])
        self.assertEqual(
            run_dict["active_optimizer_bounds"], [(-5.0, 5.0), (-5.0, 5.0)]
        )
        np.testing.assert_allclose(result["next_dofs"], [1.0, -1.0])

    def test_run_penalty_phase1_repair_first_accepts_local_violation_reduction_not_preserve_gate(
        self,
    ):
        module = self.load_module()
        anchor_hardware = {
            "success": False,
            "violations": [
                "coil_coil_min_dist 0.030000 below threshold 0.050000",
                "max_curvature 120.000000 exceeds threshold 100.000000",
            ],
            "curve_curve_min_dist": 0.03,
            "cc_dist": 0.05,
            "curve_surface_min_dist": 0.02,
            "cs_dist": 0.015,
            "surface_vessel_min_dist": 0.05,
            "ss_dist": 0.04,
            "max_curvature": 120.0,
            "curvature_threshold": 100.0,
        }
        repaired_hardware = {
            "success": False,
            "violations": [
                "coil_coil_min_dist 0.040000 below threshold 0.050000",
                "max_curvature 110.000000 exceeds threshold 100.000000",
            ],
            "curve_curve_min_dist": 0.04,
            "cc_dist": 0.05,
            "curve_surface_min_dist": 0.02,
            "cs_dist": 0.015,
            "surface_vessel_min_dist": 0.05,
            "ss_dist": 0.04,
            "max_curvature": 110.0,
            "curvature_threshold": 100.0,
        }

        repair_run_dict = {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": dict(anchor_hardware),
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
        }
        preserve_run_dict = copy.deepcopy(repair_run_dict)

        def make_callback(run_dict):
            def fake_callback(x):
                run_dict["accepted_iterations"] += 1
                run_dict["accepted_x"] = np.asarray(x, dtype=float).copy()
                run_dict["accepted_hardware_status"] = dict(repaired_hardware)
                run_dict["surface_status"] = {"success": True}

            return fake_callback

        def fake_minimize(fun, x0, **kwargs):
            kwargs["callback"](np.array([1.03, -0.97]))
            return SimpleNamespace(nit=1, success=True, message="CONVERGENCE", status=0)

        repair_result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=False,
            seed_regime="repair_first",
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=repair_run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=make_callback(repair_run_dict),
            normalize_message_fn=lambda *args, **kwargs: "phase1_ok",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        preserve_result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=True,
            seed_regime="preserve_first",
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=preserve_run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=make_callback(preserve_run_dict),
            normalize_message_fn=lambda *args, **kwargs: "phase1_ok",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        self.assertTrue(repair_result["continue_search"])
        self.assertEqual(repair_result["phase1_outcome"], "repair_local_recovery")
        self.assertEqual(repair_result["startup_local_phase_regime"], "repair_first")
        self.assertTrue(repair_result["startup_local_recovery_achieved"])
        self.assertFalse(repair_result["bridge_local_donor_ready"])
        self.assertFalse(repair_result["local_preservation_preserved_start"])
        self.assertAlmostEqual(repair_result["local_preservation_radius"], 0.015)
        self.assertFalse(preserve_result["continue_search"])
        self.assertEqual(
            preserve_result["phase1_outcome"], "preserved_start_no_safe_step"
        )
        self.assertFalse(preserve_result["startup_local_recovery_achieved"])
        self.assertFalse(preserve_result["bridge_local_donor_ready"])

    def test_run_penalty_phase1_repair_first_uses_repair_objective_not_generic_total(
        self,
    ):
        module = self.load_module()
        module.CC_WEIGHT = 101.0
        module.CS_WEIGHT = 103.0
        module.CURVATURE_WEIGHT = 107.0
        run_dict = {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": {"success": False, "violations": ["coil_coil"]},
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
        }
        captured = {}

        def fake_objective_eval(x):
            captured["repair_mode_during_eval"] = run_dict.get(
                "phase1_repair_mode_active"
            )
            return {
                "total": 999.0,
                "grad": np.array([9.0, 9.0]),
                "J_cc": 2.0,
                "dJ_cc": np.array([1.0, 0.0]),
                "J_cs": 3.0,
                "dJ_cs": np.array([0.0, 1.0]),
                "J_curvature": 4.0,
                "dJ_curvature": np.array([-1.0, 2.0]),
            }

        def fake_minimize(fun, x0, **kwargs):
            total, grad = fun(x0)
            captured["total"] = total
            captured["grad"] = grad
            return SimpleNamespace(
                nit=1, success=False, message="ABNORMAL_TERMINATION", status=2
            )

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=1,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=1,
            enable_local_preservation=False,
            seed_regime="repair_first",
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (111.0, np.array([7.0, 7.0])),
            callback_fn=lambda x: None,
            objective_eval_fn=fake_objective_eval,
            normalize_message_fn=lambda *args, **kwargs: "phase1_stop",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(
                module,
                phase1_config=module.build_phase1_config(
                    cc_weight=2.0,
                    cs_weight=3.0,
                    curvature_weight=5.0,
                ),
            ),
        )

        self.assertEqual(
            captured["total"],
            2.0 * 2.0 + 3.0 * 3.0 + 5.0 * 4.0,
        )
        np.testing.assert_allclose(
            captured["grad"],
            2.0 * np.array([1.0, 0.0])
            + 3.0 * np.array([0.0, 1.0])
            + 5.0 * np.array([-1.0, 2.0]),
        )
        self.assertTrue(captured["repair_mode_during_eval"])
        self.assertFalse(run_dict["phase1_repair_mode_active"])
        self.assertEqual(result["phase1_outcome"], "repair_first_no_local_recovery")

    def _build_repair_first_run_dict(self, *, anchor_hardware_success):
        anchor_hardware = (
            {"success": True}
            if anchor_hardware_success
            else {
                "success": False,
                "violations": [
                    "coil_coil_min_dist 0.030000 below threshold 0.050000",
                ],
                "curve_curve_min_dist": 0.03,
                "cc_dist": 0.05,
                "curve_surface_min_dist": 0.02,
                "cs_dist": 0.015,
                "surface_vessel_min_dist": 0.05,
                "ss_dist": 0.04,
                "max_curvature": 80.0,
                "curvature_threshold": 100.0,
            }
        )
        return {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": dict(anchor_hardware),
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
        }

    def test_run_penalty_phase1_repair_first_hardware_clean_donor_graduates_on_recovered_accept(
        self,
    ):
        # Regression: hardware-clean donor with anchor_repair_state==(0,0.0)
        # used to be structurally unreachable for the repair_state_improved gate.
        module = self.load_module()
        run_dict = self._build_repair_first_run_dict(anchor_hardware_success=True)

        def fake_callback(x):
            run_dict["accepted_iterations"] += 1
            run_dict["accepted_x"] = np.asarray(x, dtype=float).copy()
            run_dict["accepted_hardware_status"] = {"success": True}
            run_dict["surface_status"] = {"success": True}

        def fake_minimize(fun, x0, **kwargs):
            kwargs["callback"](np.array([1.03, -0.97]))
            return SimpleNamespace(nit=1, success=True, message="CONVERGENCE", status=0)

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=False,
            seed_regime="repair_first",
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=fake_callback,
            normalize_message_fn=lambda *args, **kwargs: "phase1_ok",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        self.assertTrue(result["continue_search"])
        self.assertEqual(result["phase1_outcome"], "repair_local_recovery_clean_anchor")
        self.assertEqual(result["startup_local_phase_regime"], "repair_first")
        self.assertTrue(result["startup_local_recovery_achieved"])
        self.assertAlmostEqual(result["local_preservation_radius"], 0.015)

    def test_run_penalty_phase1_repair_first_hardware_violating_donor_requires_repair_progress(
        self,
    ):
        # Hardware-violating donor must still demand repair_state_improved;
        # recovered_local_accept alone is not enough.
        module = self.load_module()
        run_dict = self._build_repair_first_run_dict(anchor_hardware_success=False)
        unchanged_hardware = dict(run_dict["accepted_hardware_status"])

        def fake_callback(x):
            run_dict["accepted_iterations"] += 1
            run_dict["accepted_x"] = np.asarray(x, dtype=float).copy()
            run_dict["accepted_hardware_status"] = dict(unchanged_hardware)
            run_dict["surface_status"] = {"success": True}

        def fake_minimize(fun, x0, **kwargs):
            kwargs["callback"](np.array([1.03, -0.97]))
            return SimpleNamespace(nit=1, success=True, message="CONVERGENCE", status=0)

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=False,
            seed_regime="repair_first",
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=fake_callback,
            normalize_message_fn=lambda *args, **kwargs: "phase1_reject",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        self.assertFalse(result["continue_search"])
        self.assertEqual(result["phase1_outcome"], "repair_first_no_local_recovery")
        self.assertFalse(result["startup_local_recovery_achieved"])

    def test_run_penalty_phase1_repair_first_hardware_violating_donor_graduates_when_violation_reduces(
        self,
    ):
        # Hardware-violating donor with genuine repair progress (curvature 120→110)
        # must still graduate via the original repair_local_recovery path.
        module = self.load_module()
        anchor_hardware = {
            "success": False,
            "violations": ["max_curvature 120.000000 exceeds threshold 100.000000"],
            "curve_curve_min_dist": 0.06,
            "cc_dist": 0.05,
            "curve_surface_min_dist": 0.02,
            "cs_dist": 0.015,
            "surface_vessel_min_dist": 0.05,
            "ss_dist": 0.04,
            "max_curvature": 120.0,
            "curvature_threshold": 100.0,
        }
        repaired_hardware = dict(anchor_hardware)
        repaired_hardware["max_curvature"] = 110.0
        repaired_hardware["violations"] = [
            "max_curvature 110.000000 exceeds threshold 100.000000"
        ]

        run_dict = self._build_repair_first_run_dict(anchor_hardware_success=False)
        run_dict["accepted_hardware_status"] = dict(anchor_hardware)

        def fake_callback(x):
            run_dict["accepted_iterations"] += 1
            run_dict["accepted_x"] = np.asarray(x, dtype=float).copy()
            run_dict["accepted_hardware_status"] = dict(repaired_hardware)
            run_dict["surface_status"] = {"success": True}

        def fake_minimize(fun, x0, **kwargs):
            kwargs["callback"](np.array([1.03, -0.97]))
            return SimpleNamespace(nit=1, success=True, message="CONVERGENCE", status=0)

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=False,
            seed_regime="repair_first",
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=fake_callback,
            normalize_message_fn=lambda *args, **kwargs: "phase1_ok",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        self.assertTrue(result["continue_search"])
        self.assertEqual(result["phase1_outcome"], "repair_local_recovery")
        self.assertTrue(result["startup_local_recovery_achieved"])

    def test_run_penalty_phase1_repair_first_hardware_clean_donor_fails_without_local_accept(
        self,
    ):
        # Clean anchor + step that lands on a non-refinement-ready state
        # (surface solve fails post-step) → no recovered_local_accept → no graduation.
        module = self.load_module()
        run_dict = self._build_repair_first_run_dict(anchor_hardware_success=True)

        def fake_callback(x):
            run_dict["accepted_iterations"] += 1
            run_dict["accepted_x"] = np.asarray(x, dtype=float).copy()
            run_dict["accepted_hardware_status"] = {"success": True}
            run_dict["surface_status"] = {"success": False}

        def fake_minimize(fun, x0, **kwargs):
            kwargs["callback"](np.array([1.03, -0.97]))
            return SimpleNamespace(nit=1, success=True, message="CONVERGENCE", status=0)

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=False,
            seed_regime="repair_first",
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=fake_callback,
            normalize_message_fn=lambda *args, **kwargs: "phase1_reject",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        self.assertFalse(result["continue_search"])
        self.assertEqual(result["phase1_outcome"], "repair_first_no_local_recovery")
        self.assertFalse(result["startup_local_recovery_achieved"])

    def test_run_penalty_phase1_bridge_only_requires_safe_step_for_donor_ready(self):
        module = self.load_module()
        run_dict = {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": False},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": {"success": False},
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
        }

        def fake_callback(x):
            run_dict["accepted_iterations"] += 1
            run_dict["accepted_x"] = np.asarray(x, dtype=float).copy()
            run_dict["accepted_hardware_status"] = {"success": True}
            run_dict["surface_status"] = {"success": True}

        def fake_minimize(fun, x0, **kwargs):
            kwargs["callback"](np.array([1.01, -0.99]))
            return SimpleNamespace(nit=1, success=True, message="CONVERGENCE", status=0)

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=False,
            seed_regime="bridge_only",
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=fake_callback,
            normalize_message_fn=lambda *args, **kwargs: "phase1_ok",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        self.assertTrue(result["continue_search"])
        self.assertEqual(result["phase1_outcome"], "bridge_local_donor_ready")
        self.assertEqual(result["startup_local_phase_regime"], "bridge_only")
        self.assertTrue(result["startup_local_recovery_achieved"])
        self.assertTrue(result["bridge_local_donor_ready"])

    def test_run_penalty_phase1_bridge_only_stops_without_preserved_closeout(self):
        module = self.load_module()
        run_dict = {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": False},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": {"success": False},
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
        }

        def fake_minimize(*args, **kwargs):
            return SimpleNamespace(
                nit=1, success=False, message="ABNORMAL_TERMINATION", status=2
            )

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=False,
            seed_regime="bridge_only",
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=lambda x: None,
            normalize_message_fn=lambda *args, **kwargs: "phase1_reject",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        self.assertFalse(result["continue_search"])
        self.assertEqual(result["phase1_outcome"], "bridge_only_no_local_donor")
        self.assertFalse(result["local_preservation_preserved_start"])
        self.assertEqual(result["startup_local_phase_regime"], "bridge_only")
        self.assertFalse(result["bridge_local_donor_ready"])

    def test_build_penalty_phase2_bounds_keeps_local_preservation_radius(self):
        module = self.load_module()

        bounds = module.build_penalty_phase2_bounds(
            np.array([2.0, -4.0]),
            lower_bounds=np.array([-10.0, -10.0]),
            upper_bounds=np.array([10.0, 10.0]),
            phase1_result={
                "local_preservation_used": True,
                "local_preservation_preserved_start": False,
                "local_preservation_radius": 0.05,
            },
        )

        self.assertEqual(bounds, [(1.9, 2.1), (-4.2, -3.8)])

    def test_resolve_penalty_phase2_step_norm_limit_is_opt_in(self):
        module = self.load_module()

        phase1_result = {"local_preservation_used": True}

        self.assertIsNone(
            module.resolve_penalty_phase2_step_norm_limit(
                phase1_result,
                phase1_config=module.build_phase1_config(safe_step_norm_limit=0.0),
            )
        )
        self.assertEqual(
            module.resolve_penalty_phase2_step_norm_limit(
                phase1_result,
                phase1_config=module.build_phase1_config(safe_step_norm_limit=0.005),
            ),
            0.005,
        )
        self.assertIsNone(
            module.resolve_penalty_phase2_step_norm_limit(
                {"local_preservation_used": False},
                phase1_config=module.build_phase1_config(safe_step_norm_limit=0.005),
            )
        )

    def test_combine_positive_step_norm_limits_uses_tightest_positive_limit(self):
        module = self.load_module()

        self.assertIsNone(module.combine_positive_step_norm_limits(None, 0.0))
        self.assertEqual(
            module.combine_positive_step_norm_limits(None, 0.04, 0.01),
            0.01,
        )
        with self.assertRaisesRegex(ValueError, "step norm limits must be finite"):
            module.combine_positive_step_norm_limits(float("nan"))

    def test_resolve_warm_continue_step_norm_cap_is_default_off(self):
        module = self.load_module()

        config = module.resolve_warm_continue_step_norm_cap(0.0)

        self.assertEqual(config["limit"], 0.0)
        self.assertFalse(config["enabled"])
        self.assertIsNone(config["effective_limit"])

    def test_resolve_warm_continue_step_norm_cap_requires_warm_source(self):
        module = self.load_module()

        with self.assertRaisesRegex(
            ValueError,
            "--warm-continue-step-norm-limit requires",
        ):
            module.resolve_warm_continue_step_norm_cap(0.02)

        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            module.resolve_warm_continue_step_norm_cap(float("inf"))

        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            module.resolve_warm_continue_step_norm_cap(-0.01)

    def test_resolve_warm_continue_step_norm_cap_accepts_resume_sources(self):
        module = self.load_module()

        checkpoint_config = module.resolve_warm_continue_step_norm_cap(
            0.02,
            resume_solver_checkpoint_payload={"accepted_iterations": 3},
            resume_solver_checkpoint_path="checkpoint.json",
        )
        self.assertTrue(checkpoint_config["enabled"])
        self.assertEqual(checkpoint_config["source"], "solver_checkpoint")
        self.assertEqual(checkpoint_config["source_path"], "checkpoint.json")
        self.assertEqual(checkpoint_config["effective_limit"], 0.02)
        self.assertEqual(checkpoint_config["initial_accepted_iterations"], 3)

        seed_config = module.resolve_warm_continue_step_norm_cap(
            0.03,
            seed_artifact_role=module.SEED_ARTIFACT_ROLE_SINGLE_STAGE_RESUME,
            single_stage_resume_bs_path="resume/biot_savart.json",
        )
        self.assertTrue(seed_config["enabled"])
        self.assertEqual(seed_config["source"], "single_stage_resume_seed")
        self.assertEqual(seed_config["source_path"], "resume/biot_savart.json")

        warm_start_config = module.resolve_warm_continue_step_norm_cap(
            0.04,
            warm_start_surface_stem="recovery/surf_best_feasible",
        )
        self.assertTrue(warm_start_config["enabled"])
        self.assertEqual(warm_start_config["source"], "warm_start_surface_stem")
        self.assertEqual(
            warm_start_config["source_path"],
            "recovery/surf_best_feasible",
        )

    def test_warm_continue_step_norm_cap_result_fields_stamp_inactive_defaults(self):
        module = self.load_module()

        config = module.resolve_warm_continue_step_norm_cap(0.0)
        fields = module.warm_continue_step_norm_cap_result_fields(config)

        self.assertEqual(fields["WARM_CONTINUE_STEP_NORM_LIMIT"], 0.0)
        self.assertFalse(fields["WARM_CONTINUE_STEP_NORM_CAP_ENABLED"])
        self.assertIsNone(fields["WARM_CONTINUE_STEP_NORM_CAP_SOURCE"])
        self.assertEqual(fields["WARM_CONTINUE_STEP_NORM_CAP_APPLIED_TO"], [])

    def test_evaluate_total_objective_uses_surface_weights_for_qs_and_boozer_terms(
        self,
    ):
        module = self.load_module()

        nonqs = [
            FakeAlgebraicObjective(2.0, [2.0, 0.0]),
            FakeAlgebraicObjective(6.0, [4.0, 0.0]),
        ]
        brs = [
            FakeAlgebraicObjective(10.0, [1.0, 1.0]),
            FakeAlgebraicObjective(20.0, [3.0, 3.0]),
        ]
        zero = FakeAlgebraicObjective(0.0, [0.0, 0.0])

        outer_only = module.evaluate_total_objective(
            np.array([0.0, 1.0]),
            nonqs,
            brs,
            RES_WEIGHT=2.0,
            Jiota=zero,
            IOTAS_WEIGHT=3.0,
            JCurveLength=zero,
            LENGTH_WEIGHT=4.0,
            JCurveCurve=zero,
            CC_WEIGHT=5.0,
            JCurveSurface=zero,
            CS_WEIGHT=6.0,
            JCurvature=zero,
            CURVATURE_WEIGHT=7.0,
        )
        self.assertAlmostEqual(outer_only["J_QS"], 6.0)
        self.assertAlmostEqual(outer_only["J_Boozer"], 20.0)
        np.testing.assert_allclose(outer_only["dJ_QS"], [4.0, 0.0])
        np.testing.assert_allclose(outer_only["dJ_Boozer"], [3.0, 3.0])
        self.assertAlmostEqual(outer_only["total"], 46.0)
        np.testing.assert_allclose(outer_only["grad"], [10.0, 6.0])

        ramped = module.evaluate_total_objective(
            np.array([0.5, 1.0]),
            nonqs,
            brs,
            RES_WEIGHT=2.0,
            Jiota=zero,
            IOTAS_WEIGHT=3.0,
            JCurveLength=zero,
            LENGTH_WEIGHT=4.0,
            JCurveCurve=zero,
            CC_WEIGHT=5.0,
            JCurveSurface=zero,
            CS_WEIGHT=6.0,
            JCurvature=zero,
            CURVATURE_WEIGHT=7.0,
        )
        self.assertAlmostEqual(ramped["J_QS"], (0.5 * 2.0 + 6.0) / 1.5)
        self.assertAlmostEqual(ramped["J_Boozer"], (0.5 * 10.0 + 20.0) / 1.5)
        np.testing.assert_allclose(ramped["dJ_QS"], [10.0 / 3.0, 0.0])
        np.testing.assert_allclose(ramped["dJ_Boozer"], [7.0 / 3.0, 7.0 / 3.0])
        self.assertAlmostEqual(ramped["total"], 38.0)
        np.testing.assert_allclose(ramped["grad"], [8.0, 14.0 / 3.0])

    def test_evaluate_total_objective_keeps_ramp_weights_diagnostic_with_global_objectives(
        self,
    ):
        module = self.load_module()
        module.SINGLE_STAGE_GOAL_MODE = "target"
        module.FRONTIER_GOAL_CONFIG = None

        nonqs = [
            FakeAlgebraicObjective(2.0, [2.0, 0.0]),
            FakeAlgebraicObjective(6.0, [4.0, 0.0]),
        ]
        brs = [
            FakeAlgebraicObjective(10.0, [1.0, 1.0]),
            FakeAlgebraicObjective(20.0, [3.0, 3.0]),
        ]
        global_nonqs = FakeAlgebraicObjective(100.0, [10.0, 0.0])
        global_boozer = FakeAlgebraicObjective(50.0, [0.0, 5.0])
        zero = FakeAlgebraicObjective(0.0, [0.0, 0.0])
        resolved_terms = {
            "effective_res_weight": 2.0,
            "effective_iotas_weight": 3.0,
            "effective_volume_weight": 0.0,
            "JNonQSObjective": global_nonqs,
            "JBoozerObjective": global_boozer,
            "JVolume": None,
        }

        def evaluate(weights):
            with patch.object(
                module,
                "resolve_current_surface_objective_terms",
                return_value=resolved_terms,
            ):
                return module.evaluate_total_objective(
                    np.array(weights, dtype=float),
                    nonqs,
                    brs,
                    RES_WEIGHT=999.0,
                    Jiota=zero,
                    IOTAS_WEIGHT=888.0,
                    JCurveLength=zero,
                    LENGTH_WEIGHT=4.0,
                    JCurveCurve=zero,
                    CC_WEIGHT=5.0,
                    JCurveSurface=zero,
                    CS_WEIGHT=6.0,
                    JCurvature=zero,
                    CURVATURE_WEIGHT=7.0,
                )

        outer_only = evaluate([0.0, 1.0])
        ramped = evaluate([0.5, 1.0])

        self.assertNotEqual(outer_only["J_QS"], ramped["J_QS"])
        self.assertNotEqual(outer_only["J_Boozer"], ramped["J_Boozer"])
        self.assertEqual(outer_only["J_QS_objective"], ramped["J_QS_objective"])
        self.assertEqual(
            outer_only["J_Boozer_objective"],
            ramped["J_Boozer_objective"],
        )
        self.assertEqual(outer_only["total"], ramped["total"])
        np.testing.assert_allclose(outer_only["grad"], ramped["grad"])
        np.testing.assert_allclose(ramped["surface_weights"], [0.5, 1.0])

    def test_evaluate_search_objective_uses_fast_payload_outside_frontier_mode(self):
        module = self.load_module()
        module.SINGLE_STAGE_GOAL_MODE = "target"
        module.CONSTRAINT_METHOD = "penalty"
        module.nonQSs = ["nonqs"]
        module.brs = ["brs"]
        module.RES_WEIGHT = 1.0
        module.Jiota = "jiota"
        module.IOTAS_WEIGHT = 2.0
        module.JCurveLength = "length"
        module.LENGTH_WEIGHT = 3.0
        module.JCurveCurve = "curve_curve"
        module.CC_WEIGHT = 4.0
        module.JCurveSurface = "curve_surface"
        module.CS_WEIGHT = 5.0
        module.JCurvature = "curvature"
        module.CURVATURE_WEIGHT = 6.0
        module.JShear = "shear"
        module.SHEAR_WEIGHT = 7.0
        module.JMagneticWell = "magnetic_well"
        module.MAGNETIC_WELL_WEIGHT = 8.0

        with patch.object(
            module,
            "evaluate_total_objective",
            return_value={
                "total": 7.0,
                "grad": np.array([1.0]),
                "diagnostics_included": False,
            },
        ) as evaluate_mock:
            result = module.evaluate_search_objective(np.array([1.0]))

        self.assertFalse(result["diagnostics_included"])
        self.assertFalse(evaluate_mock.call_args.kwargs["include_diagnostics"])
        self.assertEqual(evaluate_mock.call_args.kwargs["JShear"], "shear")
        self.assertEqual(evaluate_mock.call_args.kwargs["SHEAR_WEIGHT"], 7.0)
        self.assertEqual(
            evaluate_mock.call_args.kwargs["JMagneticWell"],
            "magnetic_well",
        )
        self.assertEqual(
            evaluate_mock.call_args.kwargs["MAGNETIC_WELL_WEIGHT"],
            8.0,
        )

    def test_evaluate_search_objective_penalty_includes_optional_physics_gradients(
        self,
    ):
        module = self.load_module()
        zero = FakeAlgebraicObjective(0.0, [0.0, 0.0])
        shear = FakeAlgebraicObjective(0.5, [0.25, -0.75])
        magnetic_well = FakeAlgebraicObjective(0.25, [0.5, 0.25])

        module.SINGLE_STAGE_GOAL_MODE = "target"
        module.FRONTIER_GOAL_CONFIG = None
        module.CONSTRAINT_METHOD = "penalty"
        module.nonQSs = [zero]
        module.brs = [zero]
        module.RES_WEIGHT = 0.0
        module.Jiota = zero
        module.IOTAS_WEIGHT = 0.0
        module.JCurveLength = zero
        module.LENGTH_WEIGHT = 0.0
        module.JCurveCurve = zero
        module.CC_WEIGHT = 0.0
        module.JCurveSurface = zero
        module.CS_WEIGHT = 0.0
        module.JCurvature = zero
        module.CURVATURE_WEIGHT = 0.0
        module.JPoloidalExtent = None
        module.JCurveLengthMin = None
        module.JCoilWidth = None
        module.JCurveSelfIntersect = None
        module.JResidueObjective = None
        module.JMeanSquaredCurvature = None
        module.JArclengthVariation = None
        module.JLinkingNumber = None
        module.JCoilForce = None
        module.JF = SimpleNamespace()

        module.JShear = None
        module.SHEAR_WEIGHT = 10.0
        module.JMagneticWell = None
        module.MAGNETIC_WELL_WEIGHT = 4.0
        baseline = module.evaluate_search_objective(
            np.array([1.0]),
            include_diagnostics=False,
        )

        module.JShear = shear
        module.JMagneticWell = magnetic_well
        with_terms = module.evaluate_search_objective(
            np.array([1.0]),
            include_diagnostics=False,
        )

        self.assertAlmostEqual(with_terms["total"] - baseline["total"], 6.0)
        np.testing.assert_allclose(
            with_terms["grad"] - baseline["grad"],
            [4.5, -6.5],
        )
        diagnostics = module.evaluate_search_objective(
            np.array([1.0]),
            include_diagnostics=True,
        )
        self.assertTrue(diagnostics["shear_objective_enabled"])
        self.assertAlmostEqual(diagnostics["shear_weight"], 10.0)
        self.assertAlmostEqual(diagnostics["J_shear"], 0.5)
        np.testing.assert_allclose(diagnostics["dJ_shear"], [0.25, -0.75])
        self.assertTrue(diagnostics["magnetic_well_objective_enabled"])
        self.assertAlmostEqual(diagnostics["magnetic_well_weight"], 4.0)
        self.assertAlmostEqual(diagnostics["J_magnetic_well"], 0.25)
        np.testing.assert_allclose(
            diagnostics["dJ_magnetic_well"],
            [0.5, 0.25],
        )

    def test_magnetic_well_objective_results_payload_reports_proxy_contract(self):
        module = self.load_module()
        module.MAGNETIC_WELL_WEIGHT = 4.0
        module.MAGNETIC_WELL_TARGET = -0.05
        module.JMagneticWell = SimpleNamespace(
            well_target=-0.05,
            label_inner=0.6,
            label_mid=0.8,
            label_outer=1.0,
            magnetic_well_proxy=lambda: -0.125,
        )
        search_eval = {
            "magnetic_well_objective_enabled": True,
            "magnetic_well_weight": 4.0,
            "J_magnetic_well": 0.25,
            "dJ_magnetic_well": np.array([3.0, 4.0]),
        }

        payload = module.magnetic_well_objective_results_payload(search_eval)

        self.assertTrue(payload["MAGNETIC_WELL_OBJECTIVE_ENABLED"])
        self.assertEqual(
            payload["MAGNETIC_WELL_OBJECTIVE_KIND"],
            module.MAGNETIC_WELL_OBJECTIVE_KIND,
        )
        self.assertFalse(payload["MAGNETIC_WELL_IS_FINITE_BETA_MERCIER"])
        self.assertAlmostEqual(payload["MAGNETIC_WELL_OBJECTIVE_WEIGHT"], 4.0)
        self.assertAlmostEqual(payload["MAGNETIC_WELL_OBJECTIVE_TARGET"], -0.05)
        self.assertAlmostEqual(payload["MAGNETIC_WELL_OBJECTIVE_VALUE"], 0.25)
        self.assertAlmostEqual(payload["MAGNETIC_WELL_OBJECTIVE_GRADIENT_NORM"], 5.0)
        self.assertAlmostEqual(payload["MAGNETIC_WELL_PROXY"], -0.125)
        self.assertEqual(payload["MAGNETIC_WELL_SURFACE_LABELS"], [0.6, 0.8, 1.0])
        self.assertEqual(
            payload["MAGNETIC_WELL_OBJECTIVE_PAYLOAD"]["kind"],
            module.MAGNETIC_WELL_OBJECTIVE_KIND,
        )

    def test_build_total_objective_skips_missing_volume_term(self):
        module = self.load_module()

        total = module.build_total_objective(
            FakeAlgebraicObjective(1.0, [1.0, 0.0]),
            2.0,
            FakeAlgebraicObjective(3.0, [0.0, 2.0]),
            4.0,
            FakeAlgebraicObjective(5.0, [1.0, 1.0]),
            6.0,
            None,
            8.0,
            FakeAlgebraicObjective(9.0, [0.0, 3.0]),
            10.0,
            FakeAlgebraicObjective(11.0, [1.0, -1.0]),
            12.0,
            FakeAlgebraicObjective(13.0, [0.5, 0.5]),
            14.0,
            FakeAlgebraicObjective(15.0, [2.0, -2.0]),
        )

        self.assertAlmostEqual(
            total.J(),
            1 + 2 * 3 + 4 * 5 + 8 * 9 + 10 * 11 + 12 * 13 + 14 * 15,
        )
        np.testing.assert_allclose(total.dJ(), [49.0, 0.0])

    def test_build_total_objective_includes_volume_term_when_present(self):
        module = self.load_module()

        total = module.build_total_objective(
            FakeAlgebraicObjective(1.0, [1.0, 0.0]),
            2.0,
            FakeAlgebraicObjective(3.0, [0.0, 2.0]),
            4.0,
            FakeAlgebraicObjective(5.0, [1.0, 1.0]),
            6.0,
            FakeAlgebraicObjective(7.0, [2.0, 0.0]),
            8.0,
            FakeAlgebraicObjective(9.0, [0.0, 3.0]),
            10.0,
            FakeAlgebraicObjective(11.0, [1.0, -1.0]),
            12.0,
            FakeAlgebraicObjective(13.0, [0.5, 0.5]),
            14.0,
            FakeAlgebraicObjective(15.0, [2.0, -2.0]),
        )

        self.assertAlmostEqual(
            total.J(),
            1 + 2 * 3 + 4 * 5 + 6 * 7 + 8 * 9 + 10 * 11 + 12 * 13 + 14 * 15,
        )
        np.testing.assert_allclose(total.dJ(), [61.0, 0.0])

    def test_build_total_objective_includes_length_floor_term(self):
        module = self.load_module()

        length_floor = FakeAlgebraicObjective(17.0, [4.0, 1.0])
        total = module.build_total_objective(
            FakeAlgebraicObjective(1.0, [1.0, 0.0]),
            2.0,
            FakeAlgebraicObjective(3.0, [0.0, 2.0]),
            4.0,
            FakeAlgebraicObjective(5.0, [1.0, 1.0]),
            6.0,
            None,
            8.0,
            FakeAlgebraicObjective(9.0, [0.0, 3.0]),
            10.0,
            FakeAlgebraicObjective(11.0, [1.0, -1.0]),
            12.0,
            FakeAlgebraicObjective(13.0, [0.5, 0.5]),
            14.0,
            FakeAlgebraicObjective(15.0, [2.0, -2.0]),
            JCurveLengthMin=length_floor,
        )

        self.assertAlmostEqual(
            total.J(),
            1 + 2 * 3 + 4 * 5 + 8 * 9 + 8 * 17 + 10 * 11 + 12 * 13 + 14 * 15,
        )
        np.testing.assert_allclose(total.dJ(), [81.0, 8.0])

    def test_build_total_objective_forwards_poloidal_extent_term(self):
        module = self.load_module()

        total = module.build_total_objective(
            FakeAlgebraicObjective(1.0, [1.0, 0.0]),
            2.0,
            FakeAlgebraicObjective(3.0, [0.0, 2.0]),
            4.0,
            FakeAlgebraicObjective(5.0, [1.0, 1.0]),
            6.0,
            None,
            8.0,
            FakeAlgebraicObjective(9.0, [0.0, 3.0]),
            10.0,
            FakeAlgebraicObjective(11.0, [1.0, -1.0]),
            12.0,
            FakeAlgebraicObjective(13.0, [0.5, 0.5]),
            14.0,
            FakeAlgebraicObjective(15.0, [2.0, -2.0]),
            POLOIDAL_EXTENT_WEIGHT=2.5,
            JPoloidalExtent=FakeAlgebraicObjective(17.0, [4.0, 1.0]),
        )

        self.assertAlmostEqual(
            total.J(),
            1 + 2 * 3 + 4 * 5 + 8 * 9 + 10 * 11 + 12 * 13 + 14 * 15 + 2.5 * 17,
        )
        np.testing.assert_allclose(total.dJ(), [59.0, 2.5])

    def test_build_total_objective_forwards_self_intersect_term(self):
        module = self.load_module()

        total = module.build_total_objective(
            FakeAlgebraicObjective(1.0, [1.0, 0.0]),
            2.0,
            FakeAlgebraicObjective(3.0, [0.0, 2.0]),
            4.0,
            FakeAlgebraicObjective(5.0, [1.0, 1.0]),
            6.0,
            None,
            8.0,
            FakeAlgebraicObjective(9.0, [0.0, 3.0]),
            10.0,
            FakeAlgebraicObjective(11.0, [1.0, -1.0]),
            12.0,
            FakeAlgebraicObjective(13.0, [0.5, 0.5]),
            14.0,
            FakeAlgebraicObjective(15.0, [2.0, -2.0]),
            JCurveSelfIntersect=FakeAlgebraicObjective(19.0, [0.5, -0.25]),
            SELFINT_WEIGHT=3.0,
        )

        self.assertAlmostEqual(
            total.J(),
            1 + 2 * 3 + 4 * 5 + 8 * 9 + 10 * 11 + 12 * 13 + 14 * 15 + 3 * 19,
        )
        # base length-min/volume terms zero out; only SELFINT weighted contribution
        # adds [3*0.5, 3*-0.25] = [1.5, -0.75] to the existing [49.0, 0.0].
        np.testing.assert_allclose(total.dJ(), [50.5, -0.75])

    def test_build_total_objective_omits_width_self_terms_when_objectives_missing(self):
        # ALM path keeps width/self as ALM constraints only; penalty path with
        # JCoilWidth=None and JCurveSelfIntersect=None must match the legacy
        # length-only objective so historical penalty runs see no regression.
        module = self.load_module()

        legacy = module.build_total_objective(
            FakeAlgebraicObjective(1.0, [1.0, 0.0]),
            2.0,
            FakeAlgebraicObjective(3.0, [0.0, 2.0]),
            4.0,
            FakeAlgebraicObjective(5.0, [1.0, 1.0]),
            6.0,
            None,
            8.0,
            FakeAlgebraicObjective(9.0, [0.0, 3.0]),
            10.0,
            FakeAlgebraicObjective(11.0, [1.0, -1.0]),
            12.0,
            FakeAlgebraicObjective(13.0, [0.5, 0.5]),
            14.0,
            FakeAlgebraicObjective(15.0, [2.0, -2.0]),
        )

        explicit_zero = module.build_total_objective(
            FakeAlgebraicObjective(1.0, [1.0, 0.0]),
            2.0,
            FakeAlgebraicObjective(3.0, [0.0, 2.0]),
            4.0,
            FakeAlgebraicObjective(5.0, [1.0, 1.0]),
            6.0,
            None,
            8.0,
            FakeAlgebraicObjective(9.0, [0.0, 3.0]),
            10.0,
            FakeAlgebraicObjective(11.0, [1.0, -1.0]),
            12.0,
            FakeAlgebraicObjective(13.0, [0.5, 0.5]),
            14.0,
            FakeAlgebraicObjective(15.0, [2.0, -2.0]),
            JCoilWidth=None,
            WIDTH_WEIGHT=2.0,
            JCurveSelfIntersect=None,
            SELFINT_WEIGHT=3.0,
        )

        self.assertAlmostEqual(legacy.J(), explicit_zero.J())
        np.testing.assert_allclose(legacy.dJ(), explicit_zero.dJ())

    def test_build_single_stage_objective_bundle_uses_hardware_self_intersect_distance(
        self,
    ):
        module = self.load_module()
        banana_curve = SimpleNamespace(order=8)
        outer_surface = object()
        self_objective = object()
        hardware_keepout_objective = object()
        vessel_keepout_objective = object()
        available_envelope_reward_objective = object()
        coil_force_objective = object()
        geodesic_objective = object()
        geodesic_framed_curve = object()
        recorded_self_intersect_call = {}
        recorded_hardware_keepout_call = {}
        recorded_vessel_keepout_call = {}
        recorded_available_envelope_reward_call = {}
        recorded_coil_force_call = {}
        recorded_poloidal_extent_call = {}
        recorded_width_call = {}
        recorded_geodesic_frame_call = {}
        recorded_geodesic_call = {}
        recorded_loader_call = {}

        def fake_curve_self_intersect(curve, minimum_distance, *, neighbor_skip):
            recorded_self_intersect_call.update(
                {
                    "curve": curve,
                    "minimum_distance": minimum_distance,
                    "neighbor_skip": neighbor_skip,
                }
            )
            return self_objective

        def fake_hardware_keepout(
            curves,
            points,
            minimum_distance,
            point_weight,
            winding_r0,
        ):
            recorded_hardware_keepout_call.update(
                {
                    "curves": curves,
                    "points": points,
                    "minimum_distance": minimum_distance,
                    "point_weight": point_weight,
                    "winding_r0": winding_r0,
                }
            )
            return hardware_keepout_objective

        def fake_vessel_keepout(curves, *, minimum_clearance, winding_r0):
            recorded_vessel_keepout_call.update(
                {
                    "curves": curves,
                    "minimum_clearance": minimum_clearance,
                    "winding_r0": winding_r0,
                }
            )
            return vessel_keepout_objective

        def fake_available_envelope_reward(curves, *, minimum_clearance, winding_r0):
            recorded_available_envelope_reward_call.update(
                {
                    "curves": curves,
                    "minimum_clearance": minimum_clearance,
                    "winding_r0": winding_r0,
                }
            )
            return available_envelope_reward_objective

        def fake_mean_squared_force(coil, coils, regularization):
            recorded_coil_force_call.update(
                {
                    "coil": coil,
                    "coils": coils,
                    "regularization": regularization,
                }
            )
            return coil_force_objective

        def fake_poloidal_extent(curve, major_radius, threshold):
            recorded_poloidal_extent_call.update(
                {
                    "curve": curve,
                    "major_radius": major_radius,
                    "threshold": threshold,
                }
            )
            return object()

        def fake_ellipse_width(curve, major_radius, minor_radius):
            recorded_width_call.update(
                {
                    "curve": curve,
                    "major_radius": major_radius,
                    "minor_radius": minor_radius,
                }
            )
            return object()

        def fake_surface_tangent_frame(curve, major_radius, rotation):
            recorded_geodesic_frame_call.update(
                {
                    "curve": curve,
                    "major_radius": major_radius,
                    "rotation": rotation,
                }
            )
            return geodesic_framed_curve

        def fake_geodesic_curvature(framedcurve, *, p, threshold):
            recorded_geodesic_call.update(
                {
                    "framedcurve": framedcurve,
                    "p": p,
                    "threshold": threshold,
                }
            )
            return geodesic_objective

        def fake_average_surface_objectives(terms):
            terms = list(terms)
            if terms == [self_objective]:
                return self_objective
            if terms == [hardware_keepout_objective]:
                return hardware_keepout_objective
            if terms == [geodesic_objective]:
                return geodesic_objective
            return object()

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    module,
                    "build_boozer_derived_objective_terms",
                    return_value={
                        "surface_iota_terms": ["iota"],
                        "nonQSs": ["nonqs"],
                        "brs": ["br"],
                        "boozer_objective_biot_savarts": ["bs"],
                    },
                )
            )
            for name in (
                "CurveLength",
                "Volume",
                "build_single_stage_iota_objective",
                "build_single_stage_volume_objective",
                "QuadraticPenalty",
                "CurveCurveDistance",
                "CurveSurfaceDistance",
                "LpCurveCurvature",
                "MajorRadius",
                "MinorRadius",
            ):
                stack.enter_context(patch.object(module, name, return_value=object()))
            stack.enter_context(
                patch.object(
                    module,
                    "average_surface_objectives",
                    fake_average_surface_objectives,
                )
            )
            stack.enter_context(
                patch.object(module, "PoloidalExtent", fake_poloidal_extent)
            )
            stack.enter_context(patch.object(module, "EllipseWidth", fake_ellipse_width))
            stack.enter_context(
                patch.object(
                    module,
                    "FramedCurveSurfaceTangent",
                    fake_surface_tangent_frame,
                )
            )
            stack.enter_context(
                patch.object(
                    module,
                    "CurveSurfaceGeodesicCurvature",
                    fake_geodesic_curvature,
                )
            )
            stack.enter_context(
                patch.object(
                    module,
                    "resolve_single_stage_goal_objective_terms",
                    return_value={
                        "JnonQSRatioObjective": object(),
                        "effective_res_weight": 1.0,
                        "JBoozerResidualObjective": object(),
                        "effective_iotas_weight": 1.0,
                        "effective_volume_weight": 1.0,
                    },
                )
            )
            stack.enter_context(
                patch.object(module, "CurveSelfIntersect", fake_curve_self_intersect)
            )
            keepout_points = np.array([[0.9, 0.0, 0.0], [0.91, 0.01, 0.0]])
            keepout_point_weight = 3.6e-5
            keepout_metadata = {
                "HARDWARE_KEEPOUT_GROUPS": ["shells", "limiter"],
                "HARDWARE_KEEPOUT_JSON_SHA256": "json-sha",
            }

            def fake_load_hardware_keepout(path, *, glb_path=None):
                recorded_loader_call.update({"path": path, "glb_path": glb_path})
                return (
                    keepout_points,
                    keepout_point_weight,
                    module.HARDWARE_KEEPOUT_MIN_DISTANCE_M,
                    {},
                )

            stack.enter_context(
                patch.object(
                    module,
                    "load_hardware_keepout",
                    fake_load_hardware_keepout,
                )
            )
            metadata_mock = stack.enter_context(
                patch.object(
                    module,
                    "hardware_keepout_metadata",
                    return_value=keepout_metadata,
                )
            )
            stack.enter_context(
                patch.object(module, "CurveHardwareKeepout", fake_hardware_keepout)
            )
            stack.enter_context(
                patch.object(module, "CurveVesselEnvelopeKeepout", fake_vessel_keepout)
            )
            stack.enter_context(
                patch.object(
                    module,
                    "CurveVesselAvailableEnvelopeReward",
                    fake_available_envelope_reward,
                )
            )
            stack.enter_context(
                patch.object(module, "MeanSquaredForce", fake_mean_squared_force)
            )
            build_total_mock = stack.enter_context(
                patch.object(module, "build_total_objective", return_value=object())
            )

            module.SINGLE_STAGE_HARDWARE_KEEPOUT_WEIGHT = 1.0
            module.SINGLE_STAGE_VESSEL_KEEPOUT_WEIGHT = 2.0
            module.SINGLE_STAGE_AVAILABLE_ENVELOPE_REWARD_WEIGHT = 4.0
            module.SINGLE_STAGE_VESSEL_KEEPOUT_CLEARANCE = 0.006
            module.HARDWARE_KEEPOUT_JSON_PATH = "/tmp/hardware_keepout.json"
            module.HARDWARE_KEEPOUT_GLB_PATH = "/tmp/hbt_assembly.glb"
            bundle = module.build_single_stage_objective_bundle(
                stage="full",
                surface_data=[
                    {"boozer_surface": SimpleNamespace(surface=outer_surface)}
                ],
                coils=["coil"],
                curves=["curve"],
                banana_curves=[banana_curve],
                iota_target=0.2,
                RES_WEIGHT=1.0,
                IOTAS_WEIGHT=1.0,
                LENGTH_WEIGHT=1.0,
                CC_WEIGHT=1.0,
                CC_DIST=0.05,
                CS_WEIGHT=1.0,
                CS_DIST=0.015,
                CURVATURE_WEIGHT=1.0,
                CURVATURE_THRESHOLD=20.0,
                banana_surf_radius=0.21,
                banana_surf_major_radius=0.904,
                GEODESIC_CURVATURE_WEIGHT=2.0,
                GEODESIC_CURVATURE_THRESHOLD=31.0,
                GEODESIC_CURVATURE_P=4,
                FORCE_WEIGHT=3.0,
                coil_force_regularization=0.123,
            )

        self.assertIs(recorded_self_intersect_call["curve"], banana_curve)
        self.assertEqual(recorded_poloidal_extent_call["curve"], banana_curve)
        self.assertAlmostEqual(recorded_poloidal_extent_call["major_radius"], 0.904)
        self.assertEqual(recorded_width_call["curve"], banana_curve)
        self.assertAlmostEqual(recorded_width_call["major_radius"], 0.904)
        self.assertAlmostEqual(recorded_width_call["minor_radius"], 0.21)
        self.assertIs(recorded_geodesic_frame_call["curve"], banana_curve)
        self.assertAlmostEqual(recorded_geodesic_frame_call["major_radius"], 0.904)
        self.assertEqual(recorded_geodesic_frame_call["rotation"], 0.0)
        self.assertIs(recorded_geodesic_call["framedcurve"], geodesic_framed_curve)
        self.assertEqual(recorded_geodesic_call["p"], 4)
        self.assertAlmostEqual(recorded_geodesic_call["threshold"], 31.0)
        self.assertAlmostEqual(
            recorded_self_intersect_call["minimum_distance"],
            module.BANANA_SELF_INTERSECT_MIN_DISTANCE_M,
        )
        self.assertEqual(
            recorded_self_intersect_call["neighbor_skip"],
            int(module.BANANA_SELF_INTERSECT_SKIP_ORDER_FACTOR * banana_curve.order),
        )
        self.assertEqual(
            build_total_mock.call_args.kwargs["POLOIDAL_EXTENT_WEIGHT"],
            module.SINGLE_STAGE_POLOIDAL_WEIGHT,
        )
        self.assertIs(bundle["JCurveSelfIntersect"], self_objective)
        self.assertIs(bundle["JCurveHardwareKeepout"], hardware_keepout_objective)
        self.assertEqual(
            recorded_loader_call,
            {
                "path": "/tmp/hardware_keepout.json",
                "glb_path": "/tmp/hbt_assembly.glb",
            },
        )
        metadata_mock.assert_called_once_with(
            "/tmp/hardware_keepout.json",
            glb_path="/tmp/hbt_assembly.glb",
        )
        self.assertEqual(bundle["HARDWARE_KEEPOUT_GROUP_LABELS"], ["shells", "limiter"])
        self.assertEqual(bundle["HARDWARE_KEEPOUT_METADATA"], keepout_metadata)
        self.assertEqual(recorded_hardware_keepout_call["curves"], [banana_curve])
        np.testing.assert_allclose(
            recorded_hardware_keepout_call["points"],
            keepout_points,
        )
        self.assertAlmostEqual(
            recorded_hardware_keepout_call["minimum_distance"],
            module.HARDWARE_KEEPOUT_MIN_DISTANCE_M,
        )
        self.assertAlmostEqual(
            recorded_hardware_keepout_call["point_weight"],
            keepout_point_weight,
        )
        self.assertIs(bundle["JCurveVesselEnvelopeKeepout"], vessel_keepout_objective)
        self.assertEqual(recorded_vessel_keepout_call["curves"], [banana_curve])
        self.assertAlmostEqual(
            recorded_vessel_keepout_call["minimum_clearance"],
            module.SINGLE_STAGE_VESSEL_KEEPOUT_CLEARANCE,
        )
        self.assertIs(
            bundle["JCurveAvailableEnvelopeReward"],
            available_envelope_reward_objective,
        )
        self.assertEqual(
            recorded_available_envelope_reward_call["curves"],
            [banana_curve],
        )
        self.assertAlmostEqual(
            recorded_available_envelope_reward_call["minimum_clearance"],
            module.SINGLE_STAGE_VESSEL_KEEPOUT_CLEARANCE,
        )
        self.assertIs(bundle["JCoilForce"], coil_force_objective)
        self.assertEqual(recorded_coil_force_call["coil"], "coil")
        self.assertEqual(recorded_coil_force_call["coils"], ["coil"])
        self.assertAlmostEqual(recorded_coil_force_call["regularization"], 0.123)
        self.assertIs(
            build_total_mock.call_args.kwargs["JCurveVesselEnvelopeKeepout"],
            vessel_keepout_objective,
        )
        self.assertIs(
            build_total_mock.call_args.kwargs["JCurveAvailableEnvelopeReward"],
            available_envelope_reward_objective,
        )
        self.assertEqual(
            build_total_mock.call_args.kwargs["VESSEL_KEEPOUT_WEIGHT"],
            module.SINGLE_STAGE_VESSEL_KEEPOUT_WEIGHT,
        )
        self.assertEqual(
            build_total_mock.call_args.kwargs["AVAILABLE_ENVELOPE_REWARD_WEIGHT"],
            module.SINGLE_STAGE_AVAILABLE_ENVELOPE_REWARD_WEIGHT,
        )
        self.assertIs(
            build_total_mock.call_args.kwargs["JCoilForce"],
            coil_force_objective,
        )
        self.assertEqual(build_total_mock.call_args.kwargs["FORCE_WEIGHT"], 3.0)
        self.assertIs(bundle["JGeodesicCurvature"], geodesic_objective)
        self.assertEqual(bundle["JGeodesicCurvatureTerms"], [geodesic_objective])
        self.assertIs(
            build_total_mock.call_args.kwargs["JGeodesicCurvature"],
            geodesic_objective,
        )
        self.assertEqual(
            build_total_mock.call_args.kwargs["GEODESIC_CURVATURE_WEIGHT"],
            2.0,
        )

    def test_banana_curve_order_unwraps_rotated_curve(self):
        module = self.load_module()
        banana_curve = SimpleNamespace(order=8)
        wrapped_curve = SimpleNamespace(curve=banana_curve)

        self.assertEqual(module.banana_curve_order(banana_curve), 8)
        self.assertEqual(module.banana_curve_order(wrapped_curve), 8)

    def test_build_single_stage_iota_objective_target_mode_uses_quadratic_penalty(self):
        module = self.load_module()
        surface_iota = FakeAlgebraicObjective(0.15, [1.0, -2.0])
        target_objective = object()

        with patch.object(
            module, "QuadraticPenalty", return_value=target_objective
        ) as quadratic_penalty:
            result = module.build_single_stage_iota_objective(
                surface_iota,
                0.17,
                goal_mode="target",
            )

        self.assertIs(result, target_objective)
        quadratic_penalty.assert_called_once_with(surface_iota, 0.17)

    def test_build_single_stage_iota_objective_frontier_mode_uses_bounded_reward(self):
        module = self.load_module()
        surface_iota = FakeAlgebraicObjective(0.15, [1.0, -2.0])
        frontier_goal_config = make_frontier_goal_config(module)

        result = module.build_single_stage_iota_objective(
            surface_iota,
            0.17,
            goal_mode="frontier",
            frontier_goal_config=frontier_goal_config,
        )

        self.assertAlmostEqual(result.J(), -np.tanh(1.0))
        np.testing.assert_allclose(result.dJ(), [-8.39948683, 16.79897366])

    def test_build_single_stage_volume_objective_frontier_mode_rewards_larger_volume(
        self,
    ):
        module = self.load_module()
        surface_volume = FakeAlgebraicObjective(0.12, [2.0, -1.0])
        frontier_goal_config = make_frontier_goal_config(module)

        result = module.build_single_stage_volume_objective(
            surface_volume,
            goal_mode="frontier",
            frontier_goal_config=frontier_goal_config,
        )

        self.assertAlmostEqual(result.J(), -np.tanh(2.0))
        np.testing.assert_allclose(result.dJ(), [-14.13016434, 7.06508217])

    def test_bounded_improvement_reward_partials_wrap_callable_optimizable_gradients(
        self,
    ):
        module = self.load_module()

        class DummyMetricObjective(module.Optimizable):
            def __init__(self):
                super().__init__(x0=np.array([0.0, 0.0]))

            def J(self):
                return 0.12

            def dJ(self, partials=False):
                if partials:
                    return lambda _objective: np.array([2.0, -1.0])
                return np.array([2.0, -1.0])

        metric_objective = DummyMetricObjective()
        reward = module.BoundedImprovementReward(
            metric_objective, reference=0.10, scale=0.01
        )

        partial_gradient = reward.dJ(partials=True)

        self.assertIsInstance(partial_gradient, module.Derivative)
        np.testing.assert_allclose(
            partial_gradient(metric_objective), [-14.13016497, 7.06508249]
        )

    def test_build_frontier_goal_config_derives_normalized_weights_and_trust_threshold(
        self,
    ):
        module = self.load_module()

        config = module.build_frontier_goal_config(
            initial_iota=0.15,
            initial_volume=0.10,
            target_iota=0.18,
            target_volume=0.12,
            initial_qs_objective=2.0e-4,
            initial_boozer_objective=6.0e-7,
            res_weight=1000.0,
            iotas_weight=100.0,
        )

        self.assertAlmostEqual(config.iota_reference, 0.18)
        self.assertAlmostEqual(config.iota_scale, 0.05)
        self.assertAlmostEqual(config.volume_reference, 0.12)
        self.assertAlmostEqual(config.volume_scale, 0.012)
        self.assertAlmostEqual(config.qs_reference, 2.0e-4)
        # qs_scale defaults to ``max(|initial_qs_objective|, 1e-6) * 0.25``
        # (per the iota_scale heuristic). For initial_qs_objective=2.0e-4
        # this yields 5.0e-5.
        self.assertAlmostEqual(config.qs_scale, 5.0e-5)
        self.assertAlmostEqual(config.boozer_reference, 1.0e-6)
        # boozer_scale defaults to
        # ``max(|initial_boozer_objective|, 1e-6) * 0.25 = 1e-6 * 0.25 = 2.5e-7``.
        # The post-default ``minimum=1e-6`` floor then lifts it to 1e-6.
        self.assertAlmostEqual(config.boozer_scale, 1.0e-6)
        self.assertAlmostEqual(config.boozer_trust_threshold, 1.0e-5)
        self.assertAlmostEqual(config.boozer_trust_penalty_scale, 5.0e-5)
        self.assertAlmostEqual(config.effective_boozer_weight, 1.0)
        self.assertAlmostEqual(config.effective_iota_weight, 1.0)
        self.assertAlmostEqual(config.effective_volume_weight, 1.0)

    def test_build_frontier_goal_config_volume_weight_independent_of_iota(self):
        module = self.load_module()

        config = module.build_frontier_goal_config(
            initial_iota=0.15,
            initial_volume=0.10,
            initial_qs_objective=2.0e-4,
            initial_boozer_objective=6.0e-7,
            res_weight=1000.0,
            iotas_weight=500.0,
            volume_weight=200.0,
        )

        self.assertAlmostEqual(config.effective_iota_weight, 5.0)
        self.assertAlmostEqual(config.effective_volume_weight, 2.0)

    def test_build_frontier_goal_config_volume_weight_defaults_to_iotas_weight(self):
        module = self.load_module()

        config = module.build_frontier_goal_config(
            initial_iota=0.15,
            initial_volume=0.10,
            initial_qs_objective=2.0e-4,
            initial_boozer_objective=6.0e-7,
            res_weight=1000.0,
            iotas_weight=300.0,
        )

        self.assertAlmostEqual(config.effective_iota_weight, 3.0)
        self.assertAlmostEqual(config.effective_volume_weight, 3.0)

    def test_build_frontier_goal_config_applies_reference_and_trust_overrides(self):
        module = self.load_module()

        config = module.build_frontier_goal_config(
            initial_iota=0.15,
            initial_volume=0.10,
            initial_qs_objective=2.0e-4,
            initial_boozer_objective=6.0e-7,
            res_weight=1000.0,
            iotas_weight=100.0,
            volume_weight=150.0,
            iota_reference_override=0.17,
            iota_scale_override=0.02,
            volume_reference_override=0.105,
            volume_scale_override=0.015,
            qs_reference_override=0.011,
            qs_scale_override=0.0028,
            boozer_reference_override=0.007,
            boozer_scale_override=0.0018,
            boozer_trust_threshold_override=0.009,
            boozer_trust_penalty_scale_override=0.045,
        )

        self.assertAlmostEqual(config.iota_reference, 0.17)
        self.assertAlmostEqual(config.iota_scale, 0.02)
        self.assertAlmostEqual(config.volume_reference, 0.105)
        self.assertAlmostEqual(config.volume_scale, 0.015)
        self.assertAlmostEqual(config.qs_reference, 0.011)
        self.assertAlmostEqual(config.qs_scale, 0.0028)
        self.assertAlmostEqual(config.boozer_reference, 0.007)
        self.assertAlmostEqual(config.boozer_scale, 0.0018)
        self.assertAlmostEqual(config.boozer_trust_threshold, 0.009)
        self.assertAlmostEqual(config.boozer_trust_penalty_scale, 0.045)
        self.assertAlmostEqual(config.effective_iota_weight, 1.0)
        self.assertAlmostEqual(config.effective_volume_weight, 1.5)

    def _frontier_reward_gradient_magnitude(self, module, reference, scale, metric):
        # |dJ| of the BoundedImprovementReward at a fixed metric value, used to
        # observe whether the reward still pulls (gradient alive) at that point.
        class _ConstMetric(module.Optimizable):
            def __init__(self):
                super().__init__(x0=np.array([0.0]))

            def J(self):
                return float(metric)

            def dJ(self, partials=False):
                if partials:
                    return lambda _objective: np.array([1.0])
                return np.array([1.0])

        reward = module.BoundedImprovementReward(
            _ConstMetric(), reference=reference, scale=scale
        )
        return float(np.linalg.norm(reward.dJ()))

    def test_build_frontier_goal_config_iota_ceiling_anchors_reference_and_spans_scale(
        self,
    ):
        module = self.load_module()

        config = module.build_frontier_goal_config(
            initial_iota=0.05,
            initial_volume=0.10,
            initial_qs_objective=2.0e-4,
            initial_boozer_objective=6.0e-7,
            res_weight=1000.0,
            iotas_weight=100.0,
            iota_ceiling=0.25,
        )

        # Reference is the ceiling (not the 0.05 seed); scale spans (ceiling-seed)
        # = 0.20 so the gradient stays alive across the whole seed->ceiling range.
        self.assertAlmostEqual(config.iota_reference, 0.25)
        self.assertAlmostEqual(config.iota_scale, 0.20)

    def test_build_frontier_goal_config_iota_ceiling_keeps_reward_gradient_alive_at_ceiling(
        self,
    ):
        module = self.load_module()

        seed = 0.05
        ceiling = 0.25
        legacy = module.build_frontier_goal_config(
            initial_iota=seed,
            initial_volume=0.10,
            initial_qs_objective=2.0e-4,
            initial_boozer_objective=6.0e-7,
            res_weight=1000.0,
            iotas_weight=100.0,
        )
        anchored = module.build_frontier_goal_config(
            initial_iota=seed,
            initial_volume=0.10,
            initial_qs_objective=2.0e-4,
            initial_boozer_objective=6.0e-7,
            res_weight=1000.0,
            iotas_weight=100.0,
            iota_ceiling=ceiling,
        )

        # The legacy reward anchors at the seed and saturates: its gradient is
        # nearly dead by the time iota reaches the ceiling. The ceiling-anchored
        # reward is strongest there, so it still pulls toward the limit.
        legacy_grad = self._frontier_reward_gradient_magnitude(
            module, legacy.iota_reference, legacy.iota_scale, ceiling
        )
        anchored_grad = self._frontier_reward_gradient_magnitude(
            module, anchored.iota_reference, anchored.iota_scale, ceiling
        )
        self.assertGreater(anchored_grad, 10.0 * legacy_grad)

    def test_build_frontier_goal_config_volume_ceiling_anchors_reference_and_spans_scale(
        self,
    ):
        module = self.load_module()

        config = module.build_frontier_goal_config(
            initial_iota=0.15,
            initial_volume=0.10,
            initial_qs_objective=2.0e-4,
            initial_boozer_objective=6.0e-7,
            res_weight=1000.0,
            iotas_weight=100.0,
            volume_ceiling=0.20,
        )

        self.assertAlmostEqual(config.volume_reference, 0.20)
        self.assertAlmostEqual(config.volume_scale, 0.10)

    def test_build_frontier_goal_config_ceiling_below_seed_falls_back_to_scale_floor(
        self,
    ):
        module = self.load_module()

        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            config = module.build_frontier_goal_config(
                initial_iota=0.30,
                initial_volume=0.10,
                initial_qs_objective=2.0e-4,
                initial_boozer_objective=6.0e-7,
                res_weight=1000.0,
                iotas_weight=100.0,
                iota_ceiling=0.25,
            )

        # Seed already exceeds the ceiling: reference is still the ceiling, but
        # there is no seed->ceiling span, so the scale falls back to the floor and
        # an "at or below" warning fires.
        self.assertAlmostEqual(config.iota_reference, 0.25)
        self.assertAlmostEqual(config.iota_scale, 0.05)
        self.assertIn("at or below", stdout.getvalue())

    def test_build_frontier_goal_config_small_positive_span_floors_scale_without_warning(
        self,
    ):
        module = self.load_module()

        # Seed strictly below the ceiling but within one scale-floor (span 0.04 <
        # floor 0.05): the scale is floored to 0.05 and NO "at or below" warning is
        # emitted -- the seed->ceiling span is real, just small.
        with patch("sys.stdout", new_callable=io.StringIO) as stdout:
            config = module.build_frontier_goal_config(
                initial_iota=0.21,
                initial_volume=0.10,
                initial_qs_objective=2.0e-4,
                initial_boozer_objective=6.0e-7,
                res_weight=1000.0,
                iotas_weight=100.0,
                iota_ceiling=0.25,
            )

        self.assertAlmostEqual(config.iota_reference, 0.25)
        self.assertAlmostEqual(config.iota_scale, 0.05)
        self.assertNotIn("at or below", stdout.getvalue())

    def test_build_frontier_goal_config_scale_override_wins_over_ceiling(self):
        module = self.load_module()

        config = module.build_frontier_goal_config(
            initial_iota=0.05,
            initial_volume=0.10,
            initial_qs_objective=2.0e-4,
            initial_boozer_objective=6.0e-7,
            res_weight=1000.0,
            iotas_weight=100.0,
            iota_ceiling=0.25,
            iota_scale_override=0.03,
        )

        # Reference still anchors to the ceiling, but the explicit scale override
        # beats the ceiling-derived span.
        self.assertAlmostEqual(config.iota_reference, 0.25)
        self.assertAlmostEqual(config.iota_scale, 0.03)

    def test_build_frontier_goal_config_no_ceiling_is_byte_identical(self):
        module = self.load_module()

        common = dict(
            initial_iota=0.15,
            initial_volume=0.10,
            target_iota=0.18,
            target_volume=0.12,
            initial_qs_objective=2.0e-4,
            initial_boozer_objective=6.0e-7,
            res_weight=1000.0,
            iotas_weight=100.0,
        )
        legacy = module.build_frontier_goal_config(**common)
        explicit_off = module.build_frontier_goal_config(
            **common, iota_ceiling=None, volume_ceiling=None
        )

        self.assertEqual(legacy.iota_reference, explicit_off.iota_reference)
        self.assertEqual(legacy.iota_scale, explicit_off.iota_scale)
        self.assertEqual(legacy.volume_reference, explicit_off.volume_reference)
        self.assertEqual(legacy.volume_scale, explicit_off.volume_scale)

    def test_annotate_frontier_search_eval_adds_threshold_relative_trust_penalty(self):
        module = self.load_module()
        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_GOAL_CONFIG = module.build_frontier_goal_config(
            initial_iota=0.15,
            initial_volume=0.10,
            initial_qs_objective=2.0e-4,
            initial_boozer_objective=6.0e-7,
            res_weight=1000.0,
            iotas_weight=100.0,
        )
        search_eval = {
            "total": 2.0,
            "grad": np.array([1.0, -2.0]),
            "J_Boozer": 2.5e-5,
            "dJ_Boozer": np.array([2.0e-6, -1.0e-6]),
        }

        annotated = module.annotate_frontier_search_eval(search_eval)

        self.assertEqual(annotated["frontier_base_total"], 2.0)
        self.assertFalse(annotated["frontier_trust_ok"])
        self.assertAlmostEqual(annotated["frontier_boozer_trust_threshold"], 1.0e-5)
        self.assertAlmostEqual(annotated["frontier_boozer_trust_penalty_scale"], 5.0e-5)
        self.assertAlmostEqual(annotated["frontier_boozer_trust_excess"], 1.5e-5)
        self.assertAlmostEqual(annotated["frontier_boozer_trust_excess_ratio"], 0.3)
        self.assertAlmostEqual(annotated["frontier_trust_penalty"], 0.09)
        self.assertAlmostEqual(annotated["frontier_rank_total"], 2.09)
        self.assertAlmostEqual(annotated["total"], 2.09)
        self.assertTrue(annotated["finite_eval_ok"])
        np.testing.assert_allclose(
            annotated["grad"],
            np.array([1.024, -2.012]),
        )

    def test_build_single_stage_iota_objective_rejects_invalid_goal_mode(self):
        module = self.load_module()
        surface_iota = FakeAlgebraicObjective(0.15, [1.0, -2.0])

        with self.assertRaisesRegex(ValueError, "Unsupported single-stage goal mode"):
            module.build_single_stage_iota_objective(
                surface_iota,
                0.17,
                goal_mode="not-a-mode",
            )

    def test_evaluate_surface_stack_rejects_unordered_or_too_close_surfaces(self):
        module = self.load_module()

        class _FakeSurface:
            def __init__(self, volume, points, self_intersecting=False):
                self._volume = volume
                self._points = np.asarray(points, dtype=float).reshape((-1, 1, 3))
                self._self_intersecting = self_intersecting

            def volume(self):
                return self._volume

            def gamma(self):
                return self._points

            def is_self_intersecting(self):
                return self._self_intersecting

        good_stack = [
            {
                "boozer_surface": SimpleNamespace(
                    surface=_FakeSurface(0.08, [[0.0, 0.0, 0.0]]),
                    res={"success": True, "iota": 0.12},
                )
            },
            {
                "boozer_surface": SimpleNamespace(
                    surface=_FakeSurface(0.10, [[0.4, 0.0, 0.0]]),
                    res={"success": True, "iota": 0.15},
                )
            },
        ]
        vessel = _FakeSurface(0.2, [[1.0, 0.0, 0.0]])
        good_status = module.evaluate_surface_stack(
            good_stack, vessel_surface=vessel, surface_gap_threshold=0.1
        )
        self.assertTrue(good_status["success"])
        self.assertEqual(good_status["adjacent_gaps"], [0.4])

        bad_order = [
            {
                "boozer_surface": SimpleNamespace(
                    surface=_FakeSurface(0.11, [[0.0, 0.0, 0.0]]),
                    res={"success": True, "iota": 0.12},
                )
            },
            {
                "boozer_surface": SimpleNamespace(
                    surface=_FakeSurface(0.10, [[0.4, 0.0, 0.0]]),
                    res={"success": True, "iota": 0.15},
                )
            },
        ]
        order_status = module.evaluate_surface_stack(bad_order)
        self.assertFalse(order_status["success"])
        self.assertFalse(order_status["volumes_ordered"])

        bad_gap = [
            {
                "boozer_surface": SimpleNamespace(
                    surface=_FakeSurface(0.08, [[0.0, 0.0, 0.0]]),
                    res={"success": True, "iota": 0.12},
                )
            },
            {
                "boozer_surface": SimpleNamespace(
                    surface=_FakeSurface(0.10, [[0.05, 0.0, 0.0]]),
                    res={"success": True, "iota": 0.15},
                )
            },
        ]
        gap_status = module.evaluate_surface_stack(bad_gap, surface_gap_threshold=0.1)
        self.assertFalse(gap_status["success"])
        self.assertFalse(gap_status["gap_ok"])

    def test_evaluate_surface_stack_rejects_cross_section_nesting_failure(self):
        module = self.load_module()

        class _FakeSurface:
            nfp = 5

            def __init__(self, volume, cross_section):
                self._volume = volume
                self._cross_section = np.asarray(cross_section, dtype=float)

            def volume(self):
                return self._volume

            def gamma(self):
                return self._cross_section.reshape((-1, 1, 3))

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                return self._cross_section

        inner_crossing = [
            [0.9, 0.0, -0.2],
            [1.4, 0.0, 0.0],
            [0.9, 0.0, 0.2],
            [0.6, 0.0, 0.0],
        ]
        outer_box = [
            [0.7, 0.0, -0.3],
            [1.3, 0.0, -0.3],
            [1.3, 0.0, 0.3],
            [0.7, 0.0, 0.3],
        ]
        surface_data = [
            {
                "boozer_surface": SimpleNamespace(
                    surface=_FakeSurface(0.08, inner_crossing),
                    res={"success": True, "iota": 0.12},
                )
            },
            {
                "boozer_surface": SimpleNamespace(
                    surface=_FakeSurface(0.10, outer_box),
                    res={"success": True, "iota": 0.15},
                )
            },
        ]

        status = module.evaluate_surface_stack(surface_data)
        self.assertFalse(status["success"])
        self.assertFalse(status["nesting_ok"])
        self.assertTrue(status["bad_nesting_phis"])

    def test_evaluate_surface_stack_can_skip_nesting_during_search_continuation(self):
        module = self.load_module()

        class _FakeSurface:
            nfp = 5

            def __init__(self, volume, point, cross_section):
                self._volume = volume
                self._point = np.asarray(point, dtype=float)
                self._cross_section = np.asarray(cross_section, dtype=float)

            def volume(self):
                return self._volume

            def major_radius(self):
                return in_bounds_lcfs_major_radius_m()

            def minor_radius(self):
                return in_bounds_lcfs_minor_radius_m()

            def gamma(self):
                return self._point.reshape((1, 1, 3))

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                return self._cross_section

        inner_crossing = [
            [0.9, 0.0, -0.2],
            [1.4, 0.0, 0.0],
            [0.9, 0.0, 0.2],
            [0.6, 0.0, 0.0],
        ]
        outer_box = [
            [0.7, 0.0, -0.3],
            [1.3, 0.0, -0.3],
            [1.3, 0.0, 0.3],
            [0.7, 0.0, 0.3],
        ]
        surface_data = [
            {
                "boozer_surface": SimpleNamespace(
                    surface=_FakeSurface(0.08, [0.0, 0.0, 0.0], inner_crossing),
                    res={"success": True, "iota": 0.12},
                )
            },
            {
                "boozer_surface": SimpleNamespace(
                    surface=_FakeSurface(0.10, [0.4, 0.0, 0.0], outer_box),
                    res={"success": True, "iota": 0.15},
                )
            },
        ]

        relaxed = module.evaluate_surface_stack(surface_data, enforce_nesting=False)
        self.assertTrue(relaxed["success"])
        self.assertTrue(relaxed["nesting_ok"])
        self.assertEqual(relaxed["bad_nesting_phis"], [])

        strict = module.evaluate_surface_stack(surface_data, enforce_nesting=True)
        self.assertFalse(strict["success"])
        self.assertFalse(strict["nesting_ok"])

    def test_fun_multisurface_fallback_restores_all_surface_state(self):
        module = self.load_module()
        last_J = 42.0
        last_dJ = np.array([1.0, -2.0, 3.0, -4.0, 5.0])

        class _Surface:
            nfp = 5

            def __init__(self, volume, point):
                self.x = np.array([volume])
                self._volume = volume
                self._point = np.asarray(point, dtype=float)

            def volume(self):
                return self._volume

            def major_radius(self):
                return in_bounds_lcfs_major_radius_m()

            def minor_radius(self):
                return in_bounds_lcfs_minor_radius_m()

            def gamma(self):
                return self._point.reshape((1, 1, 3))

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                return np.array(
                    [
                        [1.0, 0.0, -0.1],
                        [1.1, 0.0, 0.0],
                        [1.0, 0.0, 0.1],
                        [0.9, 0.0, 0.0],
                    ]
                )

        class _BoozerSurface:
            def __init__(self, surface, success):
                self.surface = surface
                self.res = {"success": success, "iota": TEST_IOTA, "G": TEST_G0}

            def run_code(self, iota, G):
                self.res["iota"] = iota + 0.1
                self.res["G"] = G + 0.2
                return self.res

        class _JF:
            x = np.zeros(5)

        inner = _BoozerSurface(_Surface(0.08, [0.0, 0.0, 0.0]), True)
        outer = _BoozerSurface(_Surface(0.10, [0.4, 0.0, 0.0]), False)
        module.surface_data = [
            {"boozer_surface": inner},
            {"boozer_surface": outer},
        ]
        module.run_dict = {
            "x_prev": np.zeros(5),
            "lscount": 0,
            "surface_state": {
                "sdofs": [np.array([0.08]), np.array([0.10])],
                "iota": [0.12, 0.15],
                "G": [1.0, 1.1],
            },
            "J": last_J,
            "dJ": last_dJ.copy(),
            "accepted_iterations": 0,
            "accepted_x": np.zeros(5),
        }
        module.VV = SimpleNamespace(gamma=lambda: np.array([[[1.0, 0.0, 0.0]]]))
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.JF = _JF()

        J_out, dJ_out = module.fun(np.ones(5))

        self.assertGreater(J_out, last_J)
        np.testing.assert_array_equal(dJ_out, last_dJ)
        np.testing.assert_array_equal(inner.surface.x, np.array([0.08]))
        np.testing.assert_array_equal(outer.surface.x, np.array([0.10]))
        self.assertEqual(inner.res["iota"], 0.12)
        self.assertEqual(outer.res["iota"], 0.15)

    def test_callback_multisurface_records_status_and_log(self):
        module = self.load_module()

        class _Surface:
            nfp = 5

            def __init__(self, volume, point, cross_section):
                self.x = np.array([volume])
                self._volume = volume
                self._point = np.asarray(point, dtype=float)
                self._cross_section = np.asarray(cross_section, dtype=float)

            def volume(self):
                return self._volume

            def major_radius(self):
                return in_bounds_lcfs_major_radius_m()

            def minor_radius(self):
                return in_bounds_lcfs_minor_radius_m()

            def gamma(self):
                return self._point.reshape((1, 1, 3))

            def unitnormal(self):
                return np.array([[[1.0, 0.0, 0.0]]])

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                return self._cross_section

            def save(self, path):
                self._saved_path = path

        class _ScalarObjective:
            def __init__(self, value):
                self._value = value

            def J(self):
                return self._value

            def dJ(self):
                return np.array([self._value, -self._value])

            def __add__(self, other):
                if other == 0:
                    return self
                return FakeAlgebraicObjective(
                    self._value + other.J(), self.dJ() + other.dJ()
                )

            __radd__ = __add__

            def __mul__(self, scalar):
                return FakeAlgebraicObjective(self._value * scalar, self.dJ() * scalar)

            __rmul__ = __mul__

        class _DistanceObjective(_ScalarObjective):
            def __init__(self, value, min_distance):
                super().__init__(value)
                self._min_distance = min_distance

            def shortest_distance(self):
                return self._min_distance

        class _Curve:
            def gamma(self):
                return np.array([[1.0, 0.0, 0.0]])

            def kappa(self):
                return np.array([2.0, 3.0])

        class _CurveLength:
            def J(self):
                return 4.2

        class _BS:
            def set_points(self, pts):
                self._points = pts

            def B(self):
                return np.array([[1.0, 0.0, 0.0]])

            def save(self, path):
                self._saved_path = path

        inner_cs = [
            [0.85, 0.0, -0.1],
            [1.05, 0.0, -0.1],
            [1.05, 0.0, 0.1],
            [0.85, 0.0, 0.1],
        ]
        outer_cs = [
            [0.7, 0.0, -0.3],
            [1.3, 0.0, -0.3],
            [1.3, 0.0, 0.3],
            [0.7, 0.0, 0.3],
        ]
        inner = SimpleNamespace(
            surface=_Surface(0.08, [0.0, 0.0, 0.0], inner_cs),
            res={"success": True, "iota": 0.12, "G": 1.0},
            save=lambda path: None,
        )
        outer = SimpleNamespace(
            surface=_Surface(0.10, [0.4, 0.0, 0.0], outer_cs),
            res={"success": True, "iota": 0.15, "G": 1.1},
            save=lambda path: None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            module.surface_data = [
                {
                    "name": "inner",
                    "seed_label": 0.16,
                    "target_volume": 0.08,
                    "boozer_surface": inner,
                },
                {
                    "name": "outer",
                    "seed_label": 0.20,
                    "target_volume": 0.10,
                    "boozer_surface": outer,
                },
            ]
            module.outer_surface_data = module.surface_data[-1]
            module.surface_iota_terms = [_ScalarObjective(0.12), _ScalarObjective(0.15)]
            module.nonQSs = [_ScalarObjective(0.10), _ScalarObjective(0.12)]
            module.brs = [_ScalarObjective(0.20), _ScalarObjective(0.24)]
            module.VV = SimpleNamespace(gamma=lambda: np.array([[[1.0, 0.0, 0.0]]]))
            module.SURFACE_GAP_THRESHOLD = 0.0
            module.MULTISURFACE_RAMP_ITERATIONS = 5
            module.INNER_SURFACE_INITIAL_WEIGHT = 0.0
            module.JF = _ScalarObjective(1.23)
            module.Jiota = _ScalarObjective(0.33)
            module.JCurveLength = _ScalarObjective(0.44)
            module.JCurveCurve = _DistanceObjective(0.55, 0.66)
            module.JCurveSurface = _DistanceObjective(0.77, 0.88)
            module.JCurvature = _ScalarObjective(0.99)
            module.RES_WEIGHT = 1000.0
            module.IOTAS_WEIGHT = 200.0
            module.LENGTH_WEIGHT = 1.0
            module.CC_WEIGHT = 100.0
            module.CC_DIST = 0.05
            module.CS_WEIGHT = 1.0
            module.CS_DIST = 0.02
            module.CURVATURE_WEIGHT = 0.1
            module.CURVATURE_THRESHOLD = 40.0
            module.banana_curve = _Curve()
            module.banana_curves = [module.banana_curve]
            module.curvelength = _CurveLength()
            module.CurveLength = lambda curve: _CurveLength()
            module.bs = _BS()
            module.OUT_DIR_ITER = tmpdir
            # Preserved-timeout writes during callback stamp WOUT_CONVENTION at
            # producer time; supply the TF current that the hardware snapshot
            # propagates into the payload (single_stage_banana_example.py:6657).
            module.stage2_tf_current_A = -8.0e4
            module.PRESERVED_TIMEOUT_REPLAY_CONFIG = (
                module.PreservedTimeoutReplayConfig(
                    plasma_surf_filename="wout_10x10.nc",
                    plasma_surf_path=str(SIGNED_CW_WOUT_PATH),
                    stage2_bs_path="",
                    stage2_results_path="",
                    mpol=0,
                    ntor=0,
                    nphi=0,
                    ntheta=0,
                    constraint_weight=None,
                    constraint_method=None,
                    alm_formulation=None,
                    max_iterations=None,
                    target_volume=None,
                    target_iota=None,
                )
            )
            module.run_dict = {
                "surface_state": {
                    "sdofs": [np.array([0.08]), np.array([0.10])],
                    "iota": [0.12, 0.15],
                    "G": [1.0, 1.1],
                },
                "J": 1.0,
                "dJ": np.array([1.0, -1.0]),
                "it": 1,
                "accepted_iterations": 0,
                "lscount": 0,
                "x_prev": np.zeros(2),
                "intersecting": False,
                "topology_gate_status": {
                    "enabled": False,
                    "success": True,
                    "nfieldlines": 0,
                    "survived_lines": 0,
                    "survival_fraction": 1.0,
                    "survival_threshold": 0.25,
                    "tmax": 2.0,
                    "tol": 1e-7,
                    "stop_reason_counts": {},
                    "first_exit_time": None,
                    "first_exit_angle": None,
                    "first_exit_reason": None,
                },
            }

            module.callback(np.zeros(2))

            self.assertEqual(module.run_dict["surface_status"]["adjacent_gaps"], [0.4])
            self.assertEqual(
                module.run_dict["search_surface_status"]["adjacent_gaps"], [0.4]
            )
            self.assertTrue(module.run_dict["surface_status"]["nesting_ok"])
            log_path = Path(tmpdir) / "log.txt"
            self.assertTrue(log_path.exists())
            log_text = log_path.read_text()
            self.assertIn("Adjacent surface gaps", log_text)
            self.assertIn("Surfaces nested", log_text)
            self.assertIn("Surface gate scale", log_text)

    def test_callback_tracks_relaxed_search_status_separately_from_full_status(self):
        module = self.load_module()

        class _Surface:
            nfp = 5

            def __init__(self, volume, point, cross_section):
                self.x = np.array([volume])
                self._volume = volume
                self._point = np.asarray(point, dtype=float)
                self._cross_section = np.asarray(cross_section, dtype=float)

            def volume(self):
                return self._volume

            def major_radius(self):
                return in_bounds_lcfs_major_radius_m()

            def minor_radius(self):
                return in_bounds_lcfs_minor_radius_m()

            def gamma(self):
                return self._point.reshape((1, 1, 3))

            def unitnormal(self):
                return np.array([[[1.0, 0.0, 0.0]]])

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                return self._cross_section

        class _ScalarObjective:
            def __init__(self, value):
                self._value = value

            def J(self):
                return self._value

            def dJ(self):
                return np.array([self._value, -self._value])

            def __add__(self, other):
                if other == 0:
                    return self
                return FakeAlgebraicObjective(
                    self._value + other.J(), self.dJ() + other.dJ()
                )

            __radd__ = __add__

            def __mul__(self, scalar):
                return FakeAlgebraicObjective(self._value * scalar, self.dJ() * scalar)

            __rmul__ = __mul__

        class _DistanceObjective(_ScalarObjective):
            def __init__(self, value, min_distance):
                super().__init__(value)
                self._min_distance = min_distance

            def shortest_distance(self):
                return self._min_distance

        class _Curve:
            def gamma(self):
                return np.array([[1.0, 0.0, 0.0]])

            def kappa(self):
                return np.array([2.0, 3.0])

        class _CurveLength:
            def J(self):
                return 4.2

        class _BS:
            def set_points(self, pts):
                self._points = pts

            def B(self):
                return np.array([[1.0, 0.0, 0.0]])

        inner_crossing = [
            [0.9, 0.0, -0.2],
            [1.4, 0.0, 0.0],
            [0.9, 0.0, 0.2],
            [0.6, 0.0, 0.0],
        ]
        outer_box = [
            [0.7, 0.0, -0.3],
            [1.3, 0.0, -0.3],
            [1.3, 0.0, 0.3],
            [0.7, 0.0, 0.3],
        ]
        inner = SimpleNamespace(
            surface=_Surface(0.08, [0.0, 0.0, 0.0], inner_crossing),
            res={"success": True, "iota": 0.12, "G": 1.0},
        )
        outer = SimpleNamespace(
            surface=_Surface(0.10, [0.4, 0.0, 0.0], outer_box),
            res={"success": True, "iota": 0.15, "G": 1.1},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            module.surface_data = [
                {
                    "name": "inner",
                    "seed_label": 0.16,
                    "target_volume": 0.08,
                    "boozer_surface": inner,
                },
                {
                    "name": "outer",
                    "seed_label": 0.20,
                    "target_volume": 0.10,
                    "boozer_surface": outer,
                },
            ]
            module.outer_surface_data = module.surface_data[-1]
            module.surface_iota_terms = [_ScalarObjective(0.12), _ScalarObjective(0.15)]
            module.nonQSs = [_ScalarObjective(0.10), _ScalarObjective(0.12)]
            module.brs = [_ScalarObjective(0.20), _ScalarObjective(0.24)]
            module.VV = SimpleNamespace(gamma=lambda: np.array([[[1.0, 0.0, 0.0]]]))
            module.SURFACE_GAP_THRESHOLD = 0.005
            module.MULTISURFACE_RAMP_ITERATIONS = 5
            module.INNER_SURFACE_INITIAL_WEIGHT = 0.0
            module.JF = _ScalarObjective(1.23)
            module.Jiota = _ScalarObjective(0.33)
            module.JCurveLength = _ScalarObjective(0.44)
            module.JCurveCurve = _DistanceObjective(0.55, 0.66)
            module.JCurveSurface = _DistanceObjective(0.77, 0.88)
            module.JCurvature = _ScalarObjective(0.99)
            module.RES_WEIGHT = 1000.0
            module.IOTAS_WEIGHT = 200.0
            module.LENGTH_WEIGHT = 1.0
            module.CC_WEIGHT = 100.0
            module.CC_DIST = 0.05
            module.CS_WEIGHT = 1.0
            module.CS_DIST = 0.02
            module.CURVATURE_WEIGHT = 0.1
            module.CURVATURE_THRESHOLD = 40.0
            module.banana_curve = _Curve()
            module.banana_curves = [module.banana_curve]
            module.curvelength = _CurveLength()
            module.CurveLength = lambda curve: _CurveLength()
            module.bs = _BS()
            module.OUT_DIR_ITER = tmpdir
            module.run_dict = {
                "surface_state": {
                    "sdofs": [np.array([0.08]), np.array([0.10])],
                    "iota": [0.12, 0.15],
                    "G": [1.0, 1.1],
                },
                "J": 1.0,
                "dJ": np.array([1.0, -1.0]),
                "it": 1,
                "accepted_iterations": 0,
                "lscount": 0,
                "x_prev": np.zeros(2),
                "intersecting": False,
                "curvature_overcap_boozer_evals": 7,
                "curvature_overcap_boozer_evals_this_iteration": 3,
                "topology_gate_status": {
                    "enabled": False,
                    "success": True,
                    "nfieldlines": 0,
                    "survived_lines": 0,
                    "survival_fraction": 1.0,
                    "survival_threshold": 0.25,
                    "tmax": 2.0,
                    "tol": 1e-7,
                    "stop_reason_counts": {},
                    "first_exit_time": None,
                    "first_exit_angle": None,
                    "first_exit_reason": None,
                },
            }

            module.callback(np.zeros(2))

            self.assertEqual(
                module.run_dict["curvature_overcap_boozer_evals_this_iteration"],
                0,
            )
            self.assertEqual(module.run_dict["curvature_overcap_boozer_evals"], 7)
            self.assertTrue(module.run_dict["search_surface_status"]["success"])
            self.assertTrue(module.run_dict["search_surface_status"]["nesting_ok"])
            self.assertFalse(module.run_dict["surface_status"]["success"])
            self.assertFalse(module.run_dict["surface_status"]["nesting_ok"])

    def test_finalize_surface_stack_reverts_to_last_accepted_state_when_final_endpoint_is_invalid(
        self,
    ):
        module = self.load_module()

        class _Objective:
            def __init__(self):
                self.x = np.array([0.0])

            def J(self):
                return float(self.x[0] + 10.0)

            def dJ(self):
                return np.array([self.x[0] + 1.0])

        class _Surface:
            nfp = 5

            def __init__(self, accepted_x, accepted_volume, point):
                self.x = np.array([accepted_x], dtype=float)
                self._volume = accepted_volume
                self._point = np.asarray(point, dtype=float)

            def volume(self):
                return self._volume

            def gamma(self):
                return self._point.reshape((1, 1, 3))

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                radius = self._point[0]
                return np.array(
                    [
                        [radius - 0.05, 0.0, -0.05],
                        [radius + 0.05, 0.0, -0.05],
                        [radius + 0.05, 0.0, 0.05],
                        [radius - 0.05, 0.0, 0.05],
                    ]
                )

        class _BoozerSurface:
            def __init__(self, surface, objective, valid_limit, success_iota):
                self.surface = surface
                self._objective = objective
                self._valid_limit = valid_limit
                self._success_iota = success_iota
                self.res = {"success": True, "iota": success_iota, "G": 1.0}

            def run_code(self, iota, G):
                current = float(self._objective.x[0])
                self.surface.x = np.array([current], dtype=float)
                self.surface._volume = 0.08 if self._success_iota < 0.2 else 0.10
                self.res["success"] = current <= self._valid_limit
                self.res["iota"] = self._success_iota if self.res["success"] else -1.0
                self.res["G"] = G
                return self.res

        objective = _Objective()
        inner_surface = _Surface(1.0, 0.08, [0.2, 0.0, 0.0])
        outer_surface = _Surface(1.0, 0.10, [0.6, 0.0, 0.0])
        surface_data = [
            {
                "boozer_surface": _BoozerSurface(
                    inner_surface, objective, valid_limit=1.5, success_iota=0.12
                )
            },
            {
                "boozer_surface": _BoozerSurface(
                    outer_surface, objective, valid_limit=1.5, success_iota=0.15
                )
            },
        ]
        run_state = {
            "surface_state": {
                "sdofs": [np.array([1.0]), np.array([1.0])],
                "iota": [0.12, 0.15],
                "G": [1.0, 1.1],
            },
            "accepted_x": np.array([1.0]),
            "J": 11.0,
            "dJ": np.array([2.0]),
            "intersecting": False,
        }

        status = module.finalize_surface_stack(
            np.array([2.0]), objective, surface_data, run_state
        )

        self.assertFalse(status["success"])
        np.testing.assert_allclose(objective.x, [1.0])
        np.testing.assert_allclose(run_state["accepted_x"], [1.0])
        np.testing.assert_allclose(surface_data[0]["boozer_surface"].surface.x, [1.0])
        np.testing.assert_allclose(surface_data[1]["boozer_surface"].surface.x, [1.0])
        self.assertEqual(run_state["surface_state"]["iota"], [0.12, 0.15])

    def test_minimize_alm_restores_single_stage_incumbent_state_before_finalization(
        self,
    ):
        module = self.load_module()
        alm_globals = module.minimize_alm.__globals__
        augmented_inequality_objective = alm_globals["augmented_inequality_objective"]
        grad = np.array([1.0], dtype=float)
        search_weights = np.array([1.0], dtype=float)

        class _Objective:
            def __init__(self, x):
                self.x = np.array([x], dtype=float)

            def J(self):
                return float(self.x[0])

            def dJ(self):
                return np.array([1.0], dtype=float)

        class _Surface:
            nfp = 5

            def __init__(self, x, volume, point):
                self.x = np.array([x], dtype=float)
                self._volume = volume
                self._point = np.asarray(point, dtype=float)

            def volume(self):
                return self._volume

            def gamma(self):
                return self._point.reshape((1, 1, 3))

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                radius = self._point[0]
                return np.array(
                    [
                        [radius - 0.05, 0.0, -0.05],
                        [radius + 0.05, 0.0, -0.05],
                        [radius + 0.05, 0.0, 0.05],
                        [radius - 0.05, 0.0, 0.05],
                    ]
                )

        class _BoozerSurface:
            def __init__(self, surface, objective, success_iota):
                self.surface = surface
                self._objective = objective
                self._success_iota = success_iota
                self.res = {"success": True, "iota": success_iota, "G": 1.0}

            def run_code(self, iota, G):
                current = float(self._objective.x[0])
                self.surface.x = np.array([current], dtype=float)
                self.res["success"] = True
                self.res["iota"] = self._success_iota
                self.res["G"] = G
                return self.res

        objective = _Objective(1.0)
        surface = _Surface(1.0, 0.08, [0.4, 0.0, 0.0])
        surface_data = [
            {"boozer_surface": _BoozerSurface(surface, objective, success_iota=0.12)}
        ]
        topology_status = {
            "enabled": False,
            "success": True,
            "nfieldlines": 0,
            "survived_lines": 0,
            "survival_fraction": 1.0,
            "survival_threshold": 0.25,
            "tmax": 2.0,
            "tol": 1e-7,
            "stop_reason_counts": {},
            "first_exit_time": None,
            "first_exit_angle": None,
            "first_exit_reason": None,
        }

        def make_stack_status():
            return module.evaluate_surface_stack(surface_data)

        def make_search_eval(base_value, total):
            return {
                "total": float(total),
                "base_value": float(base_value),
                "grad": grad.copy(),
            }

        run_dict = {
            "accepted_x": np.array([1.0], dtype=float),
            "surface_state": module.snapshot_surface_states(surface_data),
            "J": 9.0,
            "dJ": grad.copy(),
            "search_eval": make_search_eval(9.0, 9.0),
            "surface_status": make_stack_status(),
            "search_surface_status": make_stack_status(),
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": topology_status,
            "last_successful_eval": {"total": 999.0},
            "last_successful_eval_weights": search_weights.copy(),
        }
        settings = module.ALMSettings(
            max_outer_iterations=2,
            max_subproblem_continuations=0,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-12,
        )
        minimize_targets = iter([2.0, 3.0])

        def evaluation_profile(x):
            point = float(np.asarray(x, dtype=float)[0])
            if point >= 2.5:
                return 5.0, 0.5
            if point >= 1.5:
                return 1.0, 1.0
            return 9.0, 9.0

        def update_single_stage_state(x):
            base_value, total = evaluation_profile(x)
            x_array = np.asarray(x, dtype=float).copy()
            objective.x = x_array.copy()
            surface_data[0]["boozer_surface"].surface.x = x_array.copy()
            surface_data[0]["boozer_surface"].res["success"] = True
            surface_data[0]["boozer_surface"].res["iota"] = 0.12
            surface_data[0]["boozer_surface"].res["G"] = 1.0
            run_dict["accepted_x"] = x_array.copy()
            run_dict["surface_state"] = module.snapshot_surface_states(surface_data)
            run_dict["J"] = float(total)
            run_dict["dJ"] = grad.copy()
            run_dict["search_eval"] = make_search_eval(base_value, total)
            run_dict["surface_status"] = make_stack_status()
            run_dict["search_surface_status"] = make_stack_status()
            run_dict["accepted_hardware_status"] = {"success": True, "violations": []}
            run_dict["topology_gate_status"] = dict(topology_status)
            run_dict["last_successful_eval"] = {"total": float(total)}
            run_dict["last_successful_eval_weights"] = search_weights.copy()

        def evaluate_problem(x, multipliers, penalty):
            base_value, total = evaluation_profile(x)
            evaluation = augmented_inequality_objective(
                base_value=base_value,
                base_grad=grad.copy(),
                constraint_values=np.array([-1.0], dtype=float),
                constraint_grads=[np.zeros(1, dtype=float)],
                multipliers=np.asarray(multipliers, dtype=float),
                penalty=float(penalty),
            )
            evaluation["total"] = float(total)
            evaluation["base_value"] = float(base_value)
            evaluation["stationarity_norm"] = 1.0
            return evaluation

        def accepted_callback(x):
            update_single_stage_state(x)

        def snapshot_accepted_state():
            return module.snapshot_single_stage_incumbent_state(run_dict)

        def restore_incumbent_state(incumbent_state):
            module.restore_single_stage_incumbent_state(run_dict, incumbent_state)
            objective.x = run_dict["accepted_x"].copy()
            module.restore_surface_states(surface_data, run_dict["surface_state"])

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del x, jac, method, bounds, callback, options
            target = np.array([next(minimize_targets)], dtype=float)
            fun(target)
            return SimpleNamespace(
                x=target,
                nit=1,
                success=True,
                message="synthetic",
            )

        with patch.dict(alm_globals, {"minimize": fake_minimize}):
            result = module.minimize_alm(
                np.array([0.0], dtype=float),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
                accepted_callback=accepted_callback,
                snapshot_accepted_state_fn=snapshot_accepted_state,
                restore_incumbent_state_fn=restore_incumbent_state,
            )

        np.testing.assert_allclose(result.x, [2.0])
        np.testing.assert_allclose(run_dict["accepted_x"], [2.0])
        np.testing.assert_allclose(objective.x, [2.0])
        np.testing.assert_allclose(surface_data[0]["boozer_surface"].surface.x, [2.0])
        self.assertNotIn("last_successful_eval", run_dict)
        self.assertNotIn("last_successful_eval_weights", run_dict)

        status = module.finalize_surface_stack(
            result.x, objective, surface_data, run_dict
        )

        self.assertTrue(status["success"])
        np.testing.assert_allclose(run_dict["accepted_x"], [2.0])
        np.testing.assert_allclose(surface_data[0]["boozer_surface"].surface.x, [2.0])

    def test_collect_surface_run_metadata_serializes_multisurface_fields(self):
        module = self.load_module()
        surface_data = [
            {
                "name": "inner",
                "seed_label": 0.16,
                "target_volume": 0.08,
                "initialization_provenance": "wout_reference",
            },
            {
                "name": "outer",
                "seed_label": 0.20,
                "target_volume": 0.10,
                "initialization_provenance": "stage2_outer_seed",
            },
        ]
        run_status = {
            "self_intersections": [False, False],
            "adjacent_gaps": [0.4],
            "outer_vessel_gap": 0.6,
            "nesting_ok": True,
            "bad_nesting_phis": [],
        }
        payload = module.collect_surface_run_metadata(
            surface_data,
            run_status,
            initial_surface_volumes=[0.08, 0.10],
            initial_surface_iotas=[0.12, 0.15],
            final_surface_volumes=[0.081, 0.101],
            final_surface_iotas=[1.0 / 3.0, 0.151],
        )

        self.assertEqual(payload["SURFACE_NAMES"], ["inner", "outer"])
        self.assertEqual(payload["ADJACENT_SURFACE_GAPS"], [0.4])
        self.assertEqual(
            payload["SURFACE_INITIALIZATION_PROVENANCE"],
            ["wout_reference", "stage2_outer_seed"],
        )
        self.assertTrue(payload["SURFACES_NESTED"])
        self.assertEqual(payload["FINAL_SURFACE_VOLUMES"], [0.081, 0.101])
        self.assertTrue(payload["FINAL_INTERIOR_IOTA_NEAR_LOW_ORDER_RATIONAL"])
        self.assertEqual(
            payload["FINAL_INTERIOR_IOTA_LOW_ORDER_RATIONAL_MATCHES"],
            [
                {
                    "surface_index": 0,
                    "surface_name": "inner",
                    "iota": 1.0 / 3.0,
                    "numerator": 1,
                    "denominator": 3,
                    "rational_value": 1.0 / 3.0,
                    "abs_error": 0.0,
                }
            ],
        )
        self.assertEqual(
            payload["FINAL_INTERIOR_IOTA_LOW_ORDER_RATIONAL_CONVENTION"],
            "signed_iota=p/q; interior_surfaces_exclude_outer",
        )

    def test_collect_surface_run_metadata_ignores_outer_rational_iota(self):
        module = self.load_module()
        surface_data = [
            {
                "name": "inner",
                "seed_label": 0.16,
                "target_volume": 0.08,
                "initialization_provenance": "wout_reference",
            },
            {
                "name": "outer",
                "seed_label": 0.20,
                "target_volume": 0.10,
                "initialization_provenance": "stage2_outer_seed",
            },
        ]
        run_status = {
            "self_intersections": [False, False],
            "adjacent_gaps": [0.4],
            "outer_vessel_gap": None,
            "nesting_ok": True,
            "bad_nesting_phis": [],
        }

        payload = module.collect_surface_run_metadata(
            surface_data,
            run_status,
            initial_surface_volumes=[0.08, 0.10],
            initial_surface_iotas=[0.12, 0.15],
            final_surface_volumes=[0.081, 0.101],
            final_surface_iotas=[0.231, 1.0 / 3.0],
        )

        self.assertFalse(payload["FINAL_INTERIOR_IOTA_NEAR_LOW_ORDER_RATIONAL"])
        self.assertEqual(payload["FINAL_INTERIOR_IOTA_LOW_ORDER_RATIONAL_MATCHES"], [])


class BoozerFallbackLBFGSBTests(unittest.TestCase):
    """Issue #2: elevated-J fallback must not flush L-BFGS-B Hessian memory."""

    def test_elevated_j_stale_gradient_preserves_bfgs_memory(self):
        from scipy.optimize import minimize

        def rosenbrock(x):
            f = sum(
                100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
                for i in range(len(x) - 1)
            )
            g = np.zeros_like(x)
            for i in range(len(x) - 1):
                g[i] += -400 * x[i] * (x[i + 1] - x[i] ** 2) - 2 * (1 - x[i])
                g[i + 1] += 200 * (x[i + 1] - x[i] ** 2)
            return f, g

        rng = np.random.RandomState(42)
        x0 = rng.randn(10) * 0.5
        state = {"x_good": x0.copy(), "J": None, "dJ": None}

        def fun_with_fallback(x):
            f, g = rosenbrock(x)
            if np.linalg.norm(x - state["x_good"]) > 0.5 and state["J"] is not None:
                return state["J"] + max(abs(state["J"]), 1.0), state["dJ"].copy()
            state["J"] = f
            state["dJ"] = g.copy()
            state["x_good"] = x.copy()
            return f, g

        res = minimize(
            fun_with_fallback,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 500, "maxcor": 10},
        )

        self.assertTrue(res.success, f"L-BFGS-B did not converge: {res.message}")
        self.assertGreater(res.hess_inv.n_corrs, 0)


STAGE2_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "STAGE_2"
    / "banana_coil_solver.py"
)
STAGE2_GEOMETRY_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "banana_opt"
    / "stage2_geometry.py"
)


def load_stage2_module():
    spec = importlib.util.spec_from_file_location(
        f"banana_coil_solver_{uuid.uuid4().hex}",
        STAGE2_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_unbounded_scaled_current():
    leaf_current = SimpleNamespace(
        local_lower_bounds=np.array([-np.inf], dtype=float),
        local_upper_bounds=np.array([np.inf], dtype=float),
    )
    return leaf_current, SimpleNamespace(current_to_scale=leaf_current, scale=1.0)


def _load_segment_distance_from_source():
    """Extract the deployed segment-distance kernel from the SSOT stage2 module via AST.

    Parses the source file, extracts just the _clamp01 and segment_segment_distance
    function definitions (stripping @njit decorators), and compiles them in an
    isolated namespace. This executes the real deployed algorithm without requiring
    numba or importing the full Stage 2 workflow module.
    """
    import ast

    source = STAGE2_GEOMETRY_MODULE_PATH.read_text()
    tree = ast.parse(source)
    func_nodes = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_clamp01",
            "segment_segment_distance",
        ):
            node.decorator_list = []
            func_nodes.append(node)
    extracted = ast.Module(body=func_nodes, type_ignores=[])
    ast.fix_missing_locations(extracted)
    namespace = {"np": np}
    exec(compile(extracted, str(STAGE2_GEOMETRY_MODULE_PATH), "exec"), namespace)
    return namespace["segment_segment_distance"]


_segment_segment_distance = _load_segment_distance_from_source()


def _brute_force_segment_distance(P1, P2, Q1, Q2):
    """Reference distance via interior + 4 edge candidates on [0,1]^2."""
    u, v, w0 = P2 - P1, Q2 - Q1, P1 - Q1
    a, bv, c = np.dot(u, u), np.dot(u, v), np.dot(v, v)
    d_val, e = np.dot(u, w0), np.dot(v, w0)
    cands = []
    denom = a * c - bv * bv
    if denom > 1e-30:
        sn = (bv * e - c * d_val) / denom
        tn = (a * e - bv * d_val) / denom
        if 0.0 <= sn <= 1.0 and 0.0 <= tn <= 1.0:
            dp = w0 + sn * u - tn * v
            cands.append(np.dot(dp, dp))
    for sf in [0.0, 1.0]:
        to = max(0.0, min(1.0, (e + sf * bv) / c)) if c > 1e-30 else 0.0
        dp = w0 + sf * u - to * v
        cands.append(np.dot(dp, dp))
    for tf in [0.0, 1.0]:
        so = max(0.0, min(1.0, (tf * bv - d_val) / a)) if a > 1e-30 else 0.0
        dp = w0 + so * u - tf * v
        cands.append(np.dot(dp, dp))
    return np.sqrt(min(cands))


class SegmentDistanceTests(unittest.TestCase):
    """Issue #5/#6: segment-segment distance with Sunday/Lumelsky re-projection."""

    def _d(self, p1, p2, q1, q2):
        return _segment_segment_distance(
            np.array(p1, dtype=float),
            np.array(p2, dtype=float),
            np.array(q1, dtype=float),
            np.array(q2, dtype=float),
        )

    def test_skew_segments_reprojection(self):
        """Issue #5: buggy=1.414, correct=sqrt(1.8) after re-projection."""
        d = self._d([0, 0, 0], [2, 1, 0], [-1, 3, 0], [1, 2, 0])
        self.assertAlmostEqual(d, np.sqrt(1.8), places=10)

    def test_parallel_overlapping_segments(self):
        """Issue #6: buggy=8.06, correct=1.0 for overlapping parallel segments."""
        d = self._d([0, 0, 0], [10, 0, 0], [8, 1, 0], [20, 1, 0])
        self.assertAlmostEqual(d, 1.0, places=10)

    def test_collinear_gap(self):
        d = self._d([0, 0, 0], [1, 0, 0], [3, 0, 0], [5, 0, 0])
        self.assertAlmostEqual(d, 2.0, places=10)

    def test_perpendicular_touching(self):
        d = self._d([0, 0, 0], [1, 0, 0], [0.5, 0, 0], [0.5, 1, 0])
        self.assertAlmostEqual(d, 0.0, places=10)

    def test_point_to_segment(self):
        d = self._d([0, 2, 0], [0, 2, 0], [0, 0, 0], [1, 0, 0])
        self.assertAlmostEqual(d, 2.0, places=10)

    def test_parallel_non_overlapping(self):
        d = self._d([0, 0, 0], [3, 0, 0], [5, 1, 0], [8, 1, 0])
        self.assertAlmostEqual(d, np.sqrt(5.0), places=10)

    def test_t_shaped(self):
        d = self._d([0, 0, 0], [2, 0, 0], [1, 0.5, 0], [1, 3, 0])
        self.assertAlmostEqual(d, 0.5, places=10)

    def test_both_degenerate(self):
        d = self._d([1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6])
        self.assertAlmostEqual(d, np.linalg.norm([3, 3, 3]), places=10)

    def test_near_parallel_interior_minimum(self):
        """Near-parallel segments where the true minimum is at an interior point.

        P along x-axis, Q nearly parallel with tiny z-tilt and small y-offset.
        Endpoint projections all return sqrt(d^2 + eps^2) but the true minimum
        at (s=0.5, t=0.5) is just d (z components cancel at the midpoint).
        """
        eps = 9e-6
        d_offset = 1e-6
        d = self._d([-1, 0, 0], [1, 0, 0], [-1, d_offset, -eps], [1, d_offset, eps])
        self.assertAlmostEqual(d, d_offset, places=10)

    def test_near_parallel_brute_force(self):
        """Stress-test the near-parallel branch with 1000 adversarial pairs."""
        rng = np.random.RandomState(77777)
        PAR_EPS = 1e-10
        n_parallel = 0
        for _ in range(1000):
            base = rng.randn(3)
            base /= np.linalg.norm(base)
            P1 = rng.randn(3)
            P2 = P1 + rng.uniform(0.5, 5.0) * base
            angle = rng.uniform(1e-7, 1e-5)
            perp = rng.randn(3)
            perp -= np.dot(perp, base) * base
            perp /= np.linalg.norm(perp)
            q_dir = base + angle * perp
            q_dir /= np.linalg.norm(q_dir)
            Q1 = P1 + rng.randn(3) * rng.uniform(1e-7, 1e-4)
            Q2 = Q1 + rng.uniform(0.5, 5.0) * q_dir
            u, v, _ = P2 - P1, Q2 - Q1, P1 - Q1
            a, bv, c = np.dot(u, u), np.dot(u, v), np.dot(v, v)
            denom = a * c - bv * bv
            if denom < PAR_EPS * a * c:
                n_parallel += 1
            d_algo = _segment_segment_distance(P1, P2, Q1, Q2)
            d_brute = _brute_force_segment_distance(P1, P2, Q1, Q2)
            self.assertAlmostEqual(
                d_algo,
                d_brute,
                places=9,
                msg=f"Near-parallel mismatch: algo={d_algo}, brute={d_brute}",
            )
        self.assertGreater(
            n_parallel, 900, "Not enough pairs hit the near-parallel branch"
        )

    def test_random_brute_force(self):
        """Verify against exhaustive interior + edge search on 1000 random pairs."""
        rng = np.random.RandomState(12345)
        for _ in range(1000):
            P1, P2, Q1, Q2 = rng.randn(4, 3)
            d_algo = _segment_segment_distance(P1, P2, Q1, Q2)
            d_brute = _brute_force_segment_distance(P1, P2, Q1, Q2)
            self.assertAlmostEqual(
                d_algo,
                d_brute,
                places=9,
                msg=f"Mismatch: algo={d_algo}, brute={d_brute}",
            )


class CrossSectionNormalizationTests(unittest.TestCase):
    """Issue #8/#9: cross_section phi argument must be normalized to [0,1]."""

    PLOTTING_UTILS_PATH = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "single_stage_optimization"
        / "plotting_utils.py"
    )

    def test_plotting_utils_source_divides_by_2pi(self):
        """Verify the shared plotting_utils uses phi_slice / (2 * np.pi), not * 2 * np.pi."""
        source = self.PLOTTING_UTILS_PATH.read_text()
        self.assertIn("phi_slice / (2 * np.pi)", source)
        self.assertNotIn("phi_slice * 2 * np.pi", source)


class FtolGtolDefaultTests(unittest.TestCase):
    """ftol/gtol should be explicit tight defaults, not mpol-dependent."""

    def test_no_mpol_based_tolerance_table(self):
        """ftol/gtol are tight defaults, not mpol-dependent lookups."""
        source = EXAMPLE_MODULE_PATH.read_text()
        self.assertNotIn("ftol_by_mpol", source)
        self.assertNotIn("gtol_by_mpol", source)


class ConfinementSurrogateTests(unittest.TestCase):
    def test_topology_scorer_surrogate_emphasizes_tail_failures(self):
        topology_module = load_topology_scorer_module()

        line_metrics = [
            {"survived": True, "first_exit_time": None},
            {"survived": False, "first_exit_time": 80.0},
            {"survived": False, "first_exit_time": 20.0},
            {"survived": False, "first_exit_time": 10.0},
        ]

        surrogate = topology_module.summarize_confinement_surrogate(
            line_metrics,
            tmax=100.0,
            worst_k=2,
            early_exit_threshold=0.2,
            mean_weight=0.2,
            worst_weight=0.6,
            early_weight=0.2,
        )

        self.assertAlmostEqual(surrogate["mean_line_loss"], 0.475)
        self.assertAlmostEqual(surrogate["worst_k_line_loss"], 0.85)
        self.assertAlmostEqual(surrogate["early_exit_fraction"], 0.25)
        self.assertAlmostEqual(surrogate["confinement_loss"], 0.655)
        self.assertEqual(surrogate["confinement_surrogate_k"], 2)

    def test_checkpoint_confinement_objective_adds_weighted_loss(self):
        module = load_single_stage_example_module()

        objective = module.checkpoint_confinement_objective(
            0.125,
            {"confinement_loss": 0.4},
            3.0,
        )

        self.assertAlmostEqual(objective, 1.325)


class RunIdentityTests(unittest.TestCase):
    def _make_identity_args(self):
        return SimpleNamespace(
            boozer_stage_refinement=False,
            refinement_boozer_stage="final",
            refinement_maxiter=100,
            refinement_chunk_maxiter=20,
            refinement_max_stalled_chunks=2,
            single_stage_lane="default",
            single_stage_goal_mode="target",
            cc_dist=0.05,
            cc_weight=100.0,
            single_stage_poloidal_weight=1.0,
            single_stage_width_weight=1.0,
            single_stage_selfint_weight=1.0,
            single_stage_hardware_keepout_weight=1000.0,
            hardware_keepout_json="hardware_keepout.json",
            single_stage_vessel_keepout_weight=1000.0,
            single_stage_vessel_keepout_clearance=0.005,
            single_stage_available_envelope_reward_weight=0.0,
            single_stage_hardware_sdf_free_space_reward_weight=0.0,
            clearance_hinge_weight=0.0,
            clearance_hinge_margin_m=0.005,
            clearance_hinge_margin_profile_json=None,
            clearance_hinge_soft_min_temperature_m=0.0,
            single_stage_rational_iota_avoidance_weight=0.0,
            single_stage_rational_iota_avoidance_max_denominator=10,
            single_stage_rational_iota_avoidance_sigma=0.012,
            single_stage_iota_noble_pull_weight=0.0,
            single_stage_iota_noble_pull_lo=0.276,
            single_stage_iota_noble_pull_hi=0.281,
            single_stage_iota_pin_weight=0.0,
            single_stage_iota_pin_target=0.0860733168793427,
            single_stage_iota_pin_window=0.003,
            single_stage_geodesic_curvature_weight=0.0,
            single_stage_geodesic_curvature_threshold=39.0,
            single_stage_geodesic_curvature_p=2,
            single_stage_qa_residual_weight=0.0,
            curvature_weight=0.0001,
            curvature_threshold=40.0,
            constraint_method="penalty",
            init_only=False,
            basin_hops=0,
            basin_stepsize=0.01,
            ftol=None,
            gtol=None,
            alm_max_outer_iters=10,
            alm_penalty_init=1.0,
            alm_penalty_scale=10.0,
            alm_penalty_max=1.0e8,
            alm_feas_tol=1e-6,
            alm_stationarity_tol=1e-6,
            alm_trust_radius_init=0.0,
            alm_trust_radius_min=1e-9,
            alm_trust_radius_shrink=0.5,
            alm_trust_radius_grow=2.0,
            alm_max_inner_attempts=4,
            alm_max_subproblem_continuations=0,
            alm_distance_smoothing=1e-3,
            alm_curvature_smoothing=1e-3,
            alm_fix_signal_mismatch_guard=False,
            num_surfaces=1,
            inner_surface_ratio=0.8,
            surface_gap_threshold=0.0,
            multisurface_ramp_iterations=0,
            inner_surface_initial_weight=1.0,
            multisurface_initial_step_scale=1.0,
            multisurface_initial_step_maxiter=0,
            topology_gate_fieldlines=4,
            topology_gate_tmax=2.0,
            topology_gate_tol=1e-7,
            topology_gate_survival_threshold=0.25,
            topology_gate_penalty_scale=4.0,
            hardware_search_mode="hard",
            hardware_search_soft_iterations=0,
            curvature_traversal_band=0.0,
            curvature_traversal_eval_budget=0,
            topology_scorer_every=10,
            topology_scorer_nfieldlines=12,
            topology_scorer_tmax=50.0,
            topology_scorer_min_returns=256,
            topology_seed_mode="midplane_radial_sweep",
            topology_seed_extend_distance=0.05,
            confinement_objective_weight=0.0,
            confinement_surrogate_worst_k=3,
            confinement_surrogate_early_threshold=0.2,
            confinement_surrogate_mean_weight=0.2,
            confinement_surrogate_worst_weight=0.6,
            confinement_surrogate_early_weight=0.2,
            magnetic_well_weight=0.0,
            magnetic_well_target=0.0,
            iota_profile_weight=0.0,
            iota_profile_surface_weights=None,
            volume_profile_weight=0.0,
            iota_profile_slope=0.0,
            winding_surface_free_mpol=0,
            winding_surface_free_ntor=0,
            winding_surface_free_r0=False,
            winding_surface_free_minor=False,
        )

    def _set_residue_objective_args(
        self,
        args,
        *,
        local_difference_step=1.0e-6,
        targets_json="targets.json",
        seeds_json="seeds.json",
    ):
        args.residue_objective_weight = 0.25
        args.residue_objective_targets_json = targets_json
        args.residue_objective_seeds_json = seeds_json
        args.residue_objective_axis_r = 1.02
        args.residue_objective_axis_z = 0.01
        args.residue_objective_poloidal_orientation = 1
        args.residue_objective_radial_label_scale = 0.75
        args.residue_objective_scale = 0.5
        args.residue_objective_r_satisfied = 1.0e-5
        args.residue_objective_local_difference_step = local_difference_step
        args.residue_objective_rtol = 1.0e-10
        args.residue_objective_atol = 1.0e-12
        args.residue_objective_max_step = 0.025
        args.residue_objective_samples_per_full_torus = 384
        args.residue_objective_min_bphi_over_b = 1.0e-7
        args.residue_objective_newton_residual_tolerance = 1.0e-10
        args.residue_objective_winding_tolerance = 1.0e-8
        args.residue_objective_det_tolerance = 2.0e-6
        args.residue_objective_max_newton_iterations = 16
        args.residue_objective_max_newton_step_norm = 0.025
        return args

    def _make_identity_config(
        self,
        module,
        args,
        boozer_I=0.37,
        plasma_current_A=1850000.0,
        residue_objective=None,
    ):
        return module.make_run_identity_config(
            args,
            "stage2-seed.json",
            "final",
            0.1,
            args.constraint_method,
            0.15,
            0.15,
            boozer_I,
            plasma_current_A,
            0.22,
            80,
            80,
            None,
            residue_objective=residue_objective,
        )

    def _build_identity(
        self,
        module,
        args,
        boozer_I=0.37,
        plasma_current_A=1850000.0,
        residue_objective=None,
    ):
        return module.build_run_identity_config(
            self._make_identity_config(
                module,
                args,
                boozer_I=boozer_I,
                plasma_current_A=plasma_current_A,
                residue_objective=residue_objective,
            )
        )

    def test_run_identity_config_is_frozen(self):
        module = load_single_stage_example_module()
        args = self._make_identity_args()
        config = self._make_identity_config(module, args)

        with self.assertRaisesRegex(Exception, "cannot assign to field"):
            config.stage = "other"

    def test_run_identity_omits_default_target_goal_mode(self):
        module = load_single_stage_example_module()
        args = self._make_identity_args()
        config = self._make_identity_config(module, args)

        self.assertIsNone(config.single_stage_goal_mode)

    def test_run_identity_omits_default_banana_current_coordinate_scaling_from_hash(
        self,
    ):
        module = load_single_stage_example_module()
        args = self._make_identity_args()
        config = self._make_identity_config(module, args)
        legacy_values = [
            value
            for field, value in zip(
                fields(config),
                module.astuple(config),
            )
            if not (
                field.name
                in {
                    "single_stage_banana_current_coordinate_scaling",
                    "single_stage_banana_geometry_mode",
                    "residue_objective_weight",
                    "residue_objective_target_manifest_id",
                    "residue_objective_validation_id",
                    "residue_objective_replay_config",
                    "magnetic_well_weight",
                    "magnetic_well_target",
                    "single_stage_lane",
                    "iota_profile_surface_weights",
                    "topology_seed_mode",
                    "topology_seed_extend_distance",
                    "lcfs_constraint_mode",
                    "banana_surf_major_radius",
                    "winding_surface_free_mpol",
                    "winding_surface_free_ntor",
                    "winding_surface_free_r0",
                    "winding_surface_free_minor",
                    "single_stage_available_envelope_reward_weight",
                    "single_stage_hardware_sdf_free_space_reward_weight",
                    "single_stage_clearance_hinge_weight",
                    "single_stage_clearance_hinge_margin_m",
                    "single_stage_clearance_hinge_margin_profile_json",
                    "single_stage_clearance_hinge_soft_min_temperature_m",
                    "single_stage_rational_iota_avoidance_weight",
                    "single_stage_rational_iota_avoidance_max_denominator",
                    "single_stage_rational_iota_avoidance_sigma",
                    "single_stage_iota_noble_pull_weight",
                    "single_stage_iota_noble_pull_lo",
                    "single_stage_iota_noble_pull_hi",
                    "single_stage_iota_pin_weight",
                    "single_stage_iota_pin_target",
                    "single_stage_iota_pin_window",
                    "single_stage_geodesic_curvature_weight",
                    "single_stage_geodesic_curvature_threshold",
                    "single_stage_geodesic_curvature_p",
                    "single_stage_qa_residual_weight",
                }
                or (
                    not config.finite_build
                    and (
                        field.name == "finite_build"
                        or field.name == "requested_curvature_threshold"
                        or field.name.startswith("finitebuild_")
                    )
                )
            )
        ]

        self.assertEqual(
            module.build_run_identity_config(config),
            "|".join(str(value) for value in legacy_values),
        )

    def test_run_identity_changes_when_winding_surface_shape_requested(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        shaped_args = self._make_identity_args()
        shaped_args.winding_surface_free_mpol = 1
        shaped_args.winding_surface_free_ntor = 1

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, shaped_args),
        )

    def test_run_identity_changes_when_winding_surface_size_requested(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        shifted_args = self._make_identity_args()
        shifted_args.winding_surface_free_r0 = True
        resized_args = self._make_identity_args()
        resized_args.winding_surface_free_minor = True

        base_identity = self._build_identity(module, base_args)
        self.assertNotEqual(base_identity, self._build_identity(module, shifted_args))
        self.assertNotEqual(base_identity, self._build_identity(module, resized_args))

    def test_run_identity_changes_when_shaped_winding_surface_major_radius_changes(
        self,
    ):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        shifted_args = self._make_identity_args()
        for args in (base_args, shifted_args):
            args.winding_surface_free_mpol = 1
            args.winding_surface_free_ntor = 1
        base_args.banana_surf_major_radius = 0.903
        shifted_args.banana_surf_major_radius = 0.950

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, shifted_args),
        )

    def test_run_identity_changes_when_banana_current_coordinate_scaling_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        scaled_args = self._make_identity_args()
        scaled_args.single_stage_banana_current_mode = "independent"
        scaled_args.single_stage_banana_current_coordinate_scaling = "seed-relative"

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, scaled_args),
        )

    def test_run_identity_changes_when_banana_geometry_mode_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        materialized_args = self._make_identity_args()
        materialized_args.single_stage_banana_geometry_mode = "materialized_cws"

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, materialized_args),
        )

    def test_run_identity_uses_explicit_finite_build_threshold_state(self):
        module = load_single_stage_example_module()
        args = self._make_identity_args()
        args.finite_build = True
        args.finitebuild_numfilaments_n = module.TYPE_KK_FINITE_BUILD_NUMFILAMENTS_N
        args.finitebuild_numfilaments_b = module.TYPE_KK_FINITE_BUILD_NUMFILAMENTS_B
        args.finitebuild_gapsize_n = module.TYPE_KK_FINITE_BUILD_GAPSIZE_N_M
        args.finitebuild_gapsize_b = module.TYPE_KK_FINITE_BUILD_GAPSIZE_B_M
        args.finitebuild_rotation_order = -1
        args.finitebuild_frame = "surface_tangent"
        args.requested_curvature_threshold = 40.0
        args.curvature_threshold = 24.0
        args.finitebuild_frame_aware_threshold_enabled = True

        config = self._make_identity_config(module, args)

        self.assertAlmostEqual(config.requested_curvature_threshold, 40.0)
        self.assertAlmostEqual(config.curvature_threshold, 24.0)
        self.assertTrue(config.finitebuild_frame_aware_curvature_threshold)

    def test_run_identity_distinguishes_explicit_preserve_first_from_auto(self):
        module = load_single_stage_example_module()
        auto_args = self._make_identity_args()
        auto_args.seed_regime = "auto"
        preserve_args = self._make_identity_args()
        preserve_args.seed_regime = "preserve_first"

        auto_config = self._make_identity_config(module, auto_args)
        preserve_config = self._make_identity_config(module, preserve_args)

        self.assertIsNone(auto_config.seed_regime)
        self.assertEqual(preserve_config.seed_regime, "preserve_first")
        self.assertNotEqual(
            module.build_run_identity_config(auto_config),
            module.build_run_identity_config(preserve_config),
        )

    def test_run_identity_changes_when_only_confinement_settings_change(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        weighted_args = self._make_identity_args()
        weighted_args.confinement_objective_weight = 5.0

        base_config = self._build_identity(module, base_args)
        weighted_config = self._build_identity(module, weighted_args)

        self.assertNotEqual(base_config, weighted_config)

    def test_run_identity_changes_when_magnetic_well_settings_change(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        weighted_args = self._make_identity_args()
        weighted_args.magnetic_well_weight = 5.0
        weighted_args.magnetic_well_target = -0.05

        base_config = self._build_identity(module, base_args)
        weighted_config = self._build_identity(module, weighted_args)

        self.assertNotEqual(base_config, weighted_config)

    def test_run_identity_changes_when_residue_objective_runtime_config_changes(self):
        module = load_single_stage_example_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            targets_path = Path(tmpdir) / "targets.json"
            seeds_path = Path(tmpdir) / "seeds.json"
            targets_path.write_text('{"targets":[]}', encoding="utf-8")
            seeds_path.write_text('{"branch_seeds":[]}', encoding="utf-8")
            base_args = self._set_residue_objective_args(
                self._make_identity_args(),
                targets_json=str(targets_path),
                seeds_json=str(seeds_path),
            )
            changed_args = self._set_residue_objective_args(
                self._make_identity_args(),
                local_difference_step=2.0e-6,
                targets_json=str(targets_path),
                seeds_json=str(seeds_path),
            )
            residue_objective = SimpleNamespace(
                target_manifest_id="sha256:test-targets",
                validation_id="validation-artifact",
            )

            base_config = self._make_identity_config(
                module,
                base_args,
                residue_objective=residue_objective,
            )
            changed_config = self._make_identity_config(
                module,
                changed_args,
                residue_objective=residue_objective,
            )
            base_payload = module.residue_objective_replay_config_payload(
                base_config.residue_objective_replay_config
            )

            self.assertEqual(base_payload["target_manifest_id"], "sha256:test-targets")
            self.assertEqual(base_payload["validation_id"], "validation-artifact")
            self.assertEqual(base_payload["local_difference_step"], 1.0e-6)
            self.assertEqual(
                base_payload["targets_sha256"],
                module._optional_file_sha256(targets_path),
            )
            self.assertEqual(
                base_payload["seeds_sha256"],
                module._optional_file_sha256(seeds_path),
            )
            self.assertNotEqual(
                module.build_run_identity_config(base_config),
                module.build_run_identity_config(changed_config),
            )

            seeds_path.write_text(
                '{"branch_seeds":[{"changed":true}]}', encoding="utf-8"
            )
            changed_seed_config = self._make_identity_config(
                module,
                base_args,
                residue_objective=residue_objective,
            )
            self.assertNotEqual(
                module.build_run_identity_config(base_config),
                module.build_run_identity_config(changed_seed_config),
            )

    def test_run_identity_changes_when_goal_mode_changes(self):
        module = load_single_stage_example_module()
        target_args = self._make_identity_args()
        frontier_args = self._make_identity_args()
        frontier_args.single_stage_goal_mode = "frontier"

        target_config = self._build_identity(module, target_args)
        frontier_config = self._build_identity(module, frontier_args)

        self.assertNotEqual(target_config, frontier_config)

    def test_run_identity_changes_when_edge_iota_lane_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        edge_args = self._make_identity_args()
        edge_args.single_stage_lane = "edge_delivered_iota_lane"

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, edge_args),
        )

    def test_run_identity_changes_when_iota_profile_surface_weights_change(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        weighted_args = self._make_identity_args()
        weighted_args.iota_profile_surface_weights = "0.25,0.5,1.0,4.0"

        weighted_config = self._make_identity_config(module, weighted_args)

        self.assertEqual(
            weighted_config.iota_profile_surface_weights,
            (0.25, 0.5, 1.0, 4.0),
        )
        self.assertNotEqual(
            self._build_identity(module, base_args),
            module.build_run_identity_config(weighted_config),
        )

    def test_run_identity_changes_when_topology_seed_mode_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        edge_seed_args = self._make_identity_args()
        edge_seed_args.topology_seed_mode = "extended_surface_radial_sweep"
        edge_seed_args.topology_seed_extend_distance = 0.07

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, edge_seed_args),
        )

    def test_run_identity_changes_when_constraint_method_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        alm_args = self._make_identity_args()
        alm_args.constraint_method = "alm"

        penalty_config = self._build_identity(module, base_args)
        alm_config = self._build_identity(module, alm_args)

        self.assertNotEqual(penalty_config, alm_config)

    def test_run_identity_changes_when_single_stage_shape_weights_change(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        changed_args = self._make_identity_args()
        changed_args.single_stage_poloidal_weight = 4.0
        changed_args.single_stage_width_weight = 2.5
        changed_args.single_stage_selfint_weight = 3.5

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, changed_args),
        )

    def test_run_identity_changes_when_available_envelope_reward_weight_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        weighted_args = self._make_identity_args()
        weighted_args.single_stage_available_envelope_reward_weight = 7.5

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, weighted_args),
        )

    def test_run_identity_changes_when_hardware_sdf_reward_weight_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        weighted_args = self._make_identity_args()
        weighted_args.single_stage_hardware_sdf_free_space_reward_weight = 7.5

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, weighted_args),
        )

    def test_run_identity_changes_when_clearance_hinge_profile_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        profiled_args = self._make_identity_args()
        profiled_args.clearance_hinge_weight = 2.0
        profiled_args.clearance_hinge_margin_profile_json = "profile.json"
        profiled_args.clearance_hinge_soft_min_temperature_m = 0.002

        profiled_config = self._make_identity_config(module, profiled_args)
        profiled_identity = module.build_run_identity_config(profiled_config)

        self.assertNotEqual(
            self._build_identity(module, base_args),
            profiled_identity,
        )
        self.assertIn("profile.json", profiled_identity)
        self.assertIn("0.002", profiled_identity)

    def test_run_identity_changes_when_iota_avoidance_settings_change(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        weighted_args = self._make_identity_args()
        weighted_args.single_stage_rational_iota_avoidance_weight = 1.5
        weighted_args.single_stage_rational_iota_avoidance_max_denominator = 13
        weighted_args.single_stage_rational_iota_avoidance_sigma = 0.02

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, weighted_args),
        )

    def test_run_identity_changes_when_noble_pull_settings_change(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        weighted_args = self._make_identity_args()
        weighted_args.single_stage_iota_noble_pull_weight = 2.5
        weighted_args.single_stage_iota_noble_pull_lo = 0.276
        weighted_args.single_stage_iota_noble_pull_hi = 0.281

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, weighted_args),
        )

    def test_run_identity_changes_when_iota_pin_settings_change(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        weighted_args = self._make_identity_args()
        weighted_args.single_stage_iota_pin_weight = 1.25
        weighted_args.single_stage_iota_pin_target = 0.086
        weighted_args.single_stage_iota_pin_window = 0.002

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, weighted_args),
        )

    def test_run_identity_changes_when_geodesic_curvature_settings_change(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        weighted_args = self._make_identity_args()
        weighted_args.single_stage_geodesic_curvature_weight = 1.75
        weighted_args.single_stage_geodesic_curvature_threshold = 31.0
        weighted_args.single_stage_geodesic_curvature_p = 4

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, weighted_args),
        )

    def test_run_identity_changes_when_qa_residual_weight_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        weighted_args = self._make_identity_args()
        weighted_args.single_stage_qa_residual_weight = 3.5

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, weighted_args),
        )

    def test_run_identity_changes_when_physical_plasma_current_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()

        base_config = self._build_identity(
            module, base_args, boozer_I=0.0, plasma_current_A=0.0
        )
        physical_config = self._build_identity(
            module,
            base_args,
            boozer_I=4.0e-7 * np.pi * 8000.0,
            plasma_current_A=8000.0,
        )

        self.assertNotEqual(base_config, physical_config)

    def test_run_identity_ignores_plasma_current_input_source_when_realized_current_matches(
        self,
    ):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()

        physical_config = self._build_identity(
            module,
            base_args,
            boozer_I=4.0e-7 * np.pi * 8000.0,
            plasma_current_A=8000.0,
        )
        raw_config = self._build_identity(
            module,
            base_args,
            boozer_I=4.0e-7 * np.pi * 8000.0,
            plasma_current_A=8000.0,
        )

        self.assertEqual(physical_config, raw_config)

    def test_run_identity_changes_when_topology_gate_penalty_scale_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        changed_args = self._make_identity_args()
        changed_args.topology_gate_penalty_scale = 9.0

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, changed_args),
        )

    def test_run_identity_changes_when_hardware_search_policy_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        changed_args = self._make_identity_args()
        changed_args.hardware_search_mode = "adaptive"
        changed_args.hardware_search_soft_iterations = 3

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, changed_args),
        )

    def test_run_identity_changes_when_refinement_policy_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        changed_args = self._make_identity_args()
        changed_args.boozer_stage_refinement = True
        changed_args.refinement_maxiter = 25

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, changed_args),
        )

    def test_run_identity_changes_when_refinement_chunk_policy_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        changed_args = self._make_identity_args()
        changed_args.refinement_chunk_maxiter = 8

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, changed_args),
        )

    def test_run_identity_does_not_depend_on_module_globals(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()

        base_config = self._build_identity(module, base_args)
        module.MULTISURFACE_RAMP_ITERATIONS = 17
        module.INNER_SURFACE_INITIAL_WEIGHT = 0.25
        module.TOPOLOGY_GATE_FIELDLINES = 99
        module.TOPOLOGY_GATE_TMAX = 9.0
        module.TOPOLOGY_GATE_TOL = 1e-3
        module.TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.9
        module.TOPOLOGY_SCORER_EVERY = 77
        module.TOPOLOGY_SCORER_NFIELDLINES = 42
        module.TOPOLOGY_SCORER_TMAX = 88.0
        module.CONFINEMENT_OBJECTIVE_WEIGHT = 9.0
        module.CONFINEMENT_SURROGATE_WORST_K = 11
        module.CONFINEMENT_SURROGATE_EARLY_THRESHOLD = 0.9
        module.CONFINEMENT_SURROGATE_MEAN_WEIGHT = 0.7
        module.CONFINEMENT_SURROGATE_WORST_WEIGHT = 0.2
        module.CONFINEMENT_SURROGATE_EARLY_WEIGHT = 0.1

        self.assertEqual(base_config, self._build_identity(module, base_args))


class CurrentBaselineContractTests(unittest.TestCase):
    @staticmethod
    def _upgrade_stage2_seed_results(module, **overrides):
        hardware_contracts = load_hardware_contracts_module()
        stage2_results = {
            "MAJOR_RADIUS": module.VACUUM_VESSEL_MAJOR_RADIUS_M,
            "TOROIDAL_FLUX": 0.24,
            "banana_surf_radius": hardware_contracts.BANANA_WINDING_MINOR_RADIUS_M,
            "COIL_LENGTH": module.COIL_LENGTH_HARD_LIMIT_M,
            "CURVE_CURVE_MIN_DIST": module.COIL_COIL_MIN_DIST_M,
            "CURVE_SURFACE_MIN_DIST": module.COIL_PLASMA_MIN_DIST_M,
            "CURVATURE_THRESHOLD": module.MAX_CURVATURE_INV_M,
            "MAX_CURVATURE": module.MAX_CURVATURE_INV_M,
            "POLOIDAL_EXTENT_RAD": module.POLOIDAL_EXTENT_HALF_WIDTH_RAD,
            "POLOIDAL_EXTENT_THRESHOLD_RAD": module.POLOIDAL_EXTENT_HALF_WIDTH_RAD,
            "COIL_WIDTH": 0.10,
            "WIDTH_MIN_THRESHOLD": module.BANANA_WIDTH_MIN_M,
            "WIDTH_MAX_THRESHOLD": module.BANANA_WIDTH_MAX_M,
            "SELF_INTERSECT_PENALTY": 0.0,
            "SELF_INTERSECT_THRESHOLD": 0.0,
            "SHORTEST_SELF_DISTANCE": (
                hardware_contracts.BANANA_SELF_INTERSECT_MIN_DISTANCE_M
            ),
            "SELF_INTERSECT_MIN_DISTANCE": (
                hardware_contracts.BANANA_SELF_INTERSECT_MIN_DISTANCE_M
            ),
            "SURFACE_VESSEL_MIN_DIST": module.PLASMA_VESSEL_MIN_DIST_M,
            "BANANA_CURRENT_A": module.BANANA_CURRENT_HARD_LIMIT_A,
            "FINAL_LCFS_MAJOR_RADIUS_M": (
                hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M
            ),
            "FINAL_LCFS_MINOR_RADIUS_M": (
                hardware_contracts.TARGET_LCFS_MAX_MINOR_RADIUS_M
            ),
            "PLASMA_SURF_PATH": str(SIGNED_CW_WOUT_PATH),
            "WOUT_CONVENTION": "signed_cw",
            "WOUT_OFF_SPEC": False,
            "SEED_ROLE": "coil_seed_handoff",
            "DIAGNOSTIC_ONLY": False,
            "PRODUCTION_HANDOFF_READY": True,
            "HANDOFF_BLOCKING_GATE": None,
            "PROMOTION_READY": True,
        }
        stage2_results.update(overrides)
        return module.upgrade_legacy_stage2_artifact_results(
            stage2_results,
            known_tf_current_A=-8.0e4,
        )

    def test_stage2_seed_dir_formats_include_tf_current_segment(self):
        module = load_single_stage_example_module()
        seed_spec = module.Stage2SeedSpec(
            plasma_surf_filename="dummy.nc",
            major_radius=0.976,
            toroidal_flux=0.24,
            length_weight=0.0005,
            cc_weight=100.0,
            cc_threshold=0.05,
            curvature_weight=0.0001,
            curvature_threshold=40.0,
            banana_surf_radius=0.22,
            tf_current_A=-8.0e4,
            order=2,
        )
        local_dir = module.format_local_stage2_seed_dir(seed_spec)
        database_dir = module.format_database_stage2_seed_dir(seed_spec)

        self.assertIn("TFC=-80000", local_dir)
        self.assertIn("TFC=-80000", database_dir)
        self.assertIn("INITC=-10000", local_dir)
        self.assertIn("INITC=-10000", database_dir)

    def test_resolve_stage2_tf_current_rejects_metadata_mismatch_against_loaded_seed(
        self,
    ):
        module = load_single_stage_example_module()

        tf_coils = [
            SimpleNamespace(current=SimpleNamespace(get_value=lambda: -7.0e4)),
            SimpleNamespace(current=SimpleNamespace(get_value=lambda: -7.0e4)),
        ]

        with self.assertRaisesRegex(
            ValueError, "does not match the artifact metadata TF_CURRENT_A"
        ):
            module.resolve_stage2_tf_current_A({"TF_CURRENT_A": -8.0e4}, tf_coils)

    def test_resolve_stage2_tf_current_accepts_matching_loaded_seed_value(self):
        module = load_single_stage_example_module()

        tf_coils = [
            SimpleNamespace(current=SimpleNamespace(get_value=lambda: -8.0e4)),
            SimpleNamespace(current=SimpleNamespace(get_value=lambda: -8.0e4)),
        ]

        self.assertEqual(
            module.resolve_stage2_tf_current_A({"TF_CURRENT_A": -8.0e4}, tf_coils),
            -8.0e4,
        )

    def test_infer_uniform_tf_current_returns_none_when_coils_are_missing(self):
        module = load_single_stage_example_module()

        self.assertIsNone(module.infer_uniform_tf_current_A([]))

    def test_validate_stage2_seed_contract_rejects_missing_tf_current_metadata(self):
        module = load_single_stage_example_module()

        with self.assertRaisesRegex(
            ValueError, "missing TF_CURRENT_A even after legacy-contract upgrade"
        ):
            module.validate_stage2_seed_contract(
                {
                    "banana_surf_radius": module.BANANA_WINDING_MINOR_RADIUS_M,
                    "CURVATURE_THRESHOLD": module.MAX_CURVATURE_INV_M,
                }
            )

    def test_validate_stage2_seed_contract_accepts_upgraded_legacy_tf_current(self):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(module)

        module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_accepts_surface_vessel_clearance_at_threshold(
        self,
    ):
        import warnings

        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(
            module,
            SURFACE_VESSEL_MIN_DIST=module.PLASMA_VESSEL_MIN_DIST_M,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_accepts_diagnostic_surface_vessel_clearance_below_reference(
        self,
    ):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(
            module,
            SURFACE_VESSEL_MIN_DIST=module.PLASMA_VESSEL_MIN_DIST_M - 1.0e-3,
        )

        module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_accepts_missing_diagnostic_surface_vessel_clearance(
        self,
    ):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(module)
        stage2_results.pop("SURFACE_VESSEL_MIN_DIST", None)

        module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_rejects_missing_banana_winding_radius(self):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(module)
        stage2_results.pop("banana_surf_radius", None)

        with self.assertRaisesRegex(ValueError, "missing banana_surf_radius"):
            module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_accepts_positive_tf_current_magnitude(self):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(
            module,
            PLASMA_SURF_PATH=str(POSITIVE_CCW_WOUT_PATH),
            TF_CURRENT_A=8.0e4,
            WOUT_CONVENTION="positive_ccw",
            WOUT_OFF_SPEC=False,
        )

        module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_rejects_over_limit_tf_current(self):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(
            module,
            PLASMA_SURF_PATH=str(POSITIVE_CCW_WOUT_PATH),
            TF_CURRENT_A=8.00001e4,
            WOUT_CONVENTION="positive_ccw",
            WOUT_OFF_SPEC=False,
        )

        with self.assertRaisesRegex(ValueError, "TF current magnitude limit"):
            module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_rejects_zero_tf_current(self):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(module, TF_CURRENT_A=0.0)

        with self.assertRaisesRegex(ValueError, "finite, nonzero"):
            module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_accepts_in_vessel_banana_winding_radius_drift(
        self,
    ):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(
            module,
            banana_surf_radius=0.21,
        )

        module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_rejects_out_of_vessel_banana_winding_radius(
        self,
    ):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(
            module,
            banana_surf_radius=0.223,
        )

        with self.assertRaisesRegex(ValueError, "vacuum vessel minor radius"):
            module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_rejects_missing_curvature_threshold(self):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(module)
        stage2_results.pop("CURVATURE_THRESHOLD", None)

        with self.assertRaisesRegex(ValueError, "missing CURVATURE_THRESHOLD"):
            module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_accepts_curvature_threshold_below_ceiling(
        self,
    ):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(
            module,
            CURVATURE_THRESHOLD=50.0,
        )

        module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_rejects_curvature_threshold_above_ceiling(
        self,
    ):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(
            module,
            CURVATURE_THRESHOLD=100.1,
        )

        with self.assertRaisesRegex(ValueError, "curvature threshold exceeds"):
            module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_rejects_nan_curvature_threshold(self):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(
            module,
            CURVATURE_THRESHOLD=float("nan"),
        )

        with self.assertRaisesRegex(ValueError, "CURVATURE_THRESHOLD must be finite"):
            module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_accepts_missing_diagnostic_poloidal_extent_threshold(
        self,
    ):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(module)
        stage2_results.pop("POLOIDAL_EXTENT_THRESHOLD_RAD", None)

        module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_accepts_missing_diagnostic_full_contract_metric(
        self,
    ):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(module)
        stage2_results.pop("COIL_LENGTH", None)

        module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_accepts_lcfs_major_radius_telemetry(self):
        module = load_single_stage_example_module()
        hardware_contracts = load_hardware_contracts_module()
        stage2_results = self._upgrade_stage2_seed_results(
            module,
            FINAL_LCFS_MAJOR_RADIUS_M=(
                hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M + 1.0e-3
            ),
        )

        module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_accepts_banana_current_telemetry(self):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(
            module,
            BANANA_CURRENT_A=module.BANANA_CURRENT_HARD_LIMIT_A + 1.0,
        )

        module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_contract_accepts_poloidal_extent_threshold_telemetry(
        self,
    ):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(
            module,
            POLOIDAL_EXTENT_THRESHOLD_RAD=module.POLOIDAL_EXTENT_HALF_WIDTH_RAD
            + 1.0e-3,
        )

        module.validate_stage2_seed_contract(stage2_results)

    def test_validate_stage2_seed_bootability_contract_rejects_noncoil_seed_handoff(
        self,
    ):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(
            module,
            BOOZER_BOOTABLE=False,
            IOTA_FEASIBLE=False,
            BOOTABILITY_REASON="self_intersection",
            BOOTABILITY_SOLVED_IOTA=-1.0e-5,
            BOOTABILITY_TARGET_IOTA=0.16,
        )

        with self.assertRaisesRegex(ValueError, "not single-stage bootable"):
            module.validate_stage2_seed_bootability_contract(stage2_results)

    def test_validate_stage2_seed_bootability_contract_rejects_iota_miss_handoff(self):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(
            module,
            BOOZER_BOOTABLE=True,
            BOOZER_TRUSTED=True,
            IOTA_NEAR_TARGET=False,
            IOTA_FEASIBLE=False,
            BOOTABILITY_REASON="ok",
            BOOTABILITY_SOLVED_IOTA=0.12,
            BOOTABILITY_TARGET_IOTA=0.16,
        )

        with self.assertRaisesRegex(ValueError, "IOTA_NEAR_TARGET=False"):
            module.validate_stage2_seed_bootability_contract(stage2_results)

    def test_validate_stage2_seed_bootability_contract_accepts_handoff_by_facts(self):
        module = load_single_stage_example_module()
        stage2_results = self._upgrade_stage2_seed_results(
            module,
            BOOZER_BOOTABLE=True,
            BOOZER_TRUSTED=True,
            IOTA_NEAR_TARGET=True,
            IOTA_FEASIBLE=True,
            BOOTABILITY_REASON="ok",
            BOOTABILITY_SOLVED_IOTA=0.16,
            BOOTABILITY_TARGET_IOTA=0.16,
        )

        module.validate_stage2_seed_bootability_contract(stage2_results)

    def test_resolve_single_stage_banana_surf_radius_defaults_to_loaded_artifact(self):
        module = load_single_stage_example_module()

        self.assertEqual(
            module.resolve_single_stage_banana_surf_radius(
                {"banana_surf_radius": 0.22},
                None,
            ),
            0.22,
        )

    def test_resolve_single_stage_banana_surf_radius_rejects_cli_override_mismatch(
        self,
    ):
        module = load_single_stage_example_module()

        with self.assertRaisesRegex(
            ValueError, "must match the loaded Stage 2 artifact radius 0.220000 m"
        ):
            module.resolve_single_stage_banana_surf_radius(
                {"banana_surf_radius": 0.22},
                0.21,
            )

    def test_resolve_stage2_num_tf_coils_prefers_recorded_artifact_count(self):
        module = load_single_stage_example_module()
        stage2_results = {"NUM_TF_COILS": 20}

        self.assertEqual(
            module.resolve_stage2_num_tf_coils(
                stage2_results, requested_num_tf_coils=20
            ),
            20,
        )

    def test_resolve_stage2_num_tf_coils_rejects_cli_mismatch(self):
        module = load_single_stage_example_module()
        stage2_results = {"NUM_TF_COILS": 18}

        with self.assertRaisesRegex(ValueError, "NUM_TF_COILS=18.*--num-tf-coils=20"):
            module.resolve_stage2_num_tf_coils(
                stage2_results, requested_num_tf_coils=20
            )

    def test_validate_loaded_stage2_coils_partition_rejects_too_few_loaded_coils(self):
        module = load_single_stage_example_module()

        with self.assertRaisesRegex(ValueError, "has only 19 coils.*NUM_TF_COILS=20"):
            module.validate_loaded_stage2_coils_partition(
                [object()] * 19,
                stage2_results={"NUM_TF_COILS": 20},
                requested_num_tf_coils=20,
            )

    def test_validate_loaded_stage2_coils_partition_rejects_missing_banana_coils(self):
        module = load_single_stage_example_module()

        with self.assertRaisesRegex(ValueError, "leaving no banana coils"):
            module.validate_loaded_stage2_coils_partition(
                [object()] * 20,
                stage2_results={"NUM_TF_COILS": 20},
                requested_num_tf_coils=20,
            )

    def test_validate_loaded_seed_current_source_contract_rejects_resume_override(
        self,
    ):
        module = load_single_stage_example_module()

        with self.assertRaisesRegex(
            ValueError,
            "cannot retarget physical plasma current",
        ):
            module.validate_loaded_seed_current_source_contract(
                finite_current_mode="wataru_proxy_field",
                effective_current_mode="wataru_proxy_field",
                plasma_current_A=-400.0,
                plasma_current_input_source="physical_A",
                stage2_results={
                    "FINITE_CURRENT_MODE": "wataru_proxy_field",
                    "PROXY_PLASMA_CURRENT_A": 0.0,
                    "VF_CURRENT_A": 0.0,
                },
                coil_partitions=SimpleNamespace(
                    num_proxy_coils=0,
                    num_vf_coils=0,
                ),
            )

    def test_build_stage2_bs_path_prefers_current_penalty_dir(self):
        module = load_single_stage_example_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "outputs-demo.nc"
            current_dir = (
                outputs_dir
                / "R0=0.976-s=0.24-LW=0.0005-CCW=100-CCT=0.05-CW=0.0001-CT=40-SR=0.220-INITC=-10000-MAXC=16000-TFC=-80000-Order=2-CM=penalty"
            )
            current_dir.mkdir(parents=True)
            expected_path = current_dir / "biot_savart_opt.json"
            expected_path.write_text("{}", encoding="utf-8")

            args = SimpleNamespace(
                stage2_bs_path=None,
                stage2_source="local",
                local_stage2_root=tmpdir,
                database_stage2_root="/unused",
                plasma_surf_filename="demo.nc",
                stage2_seed_major_radius=0.976,
                stage2_seed_toroidal_flux=0.24,
                stage2_seed_length_weight=0.0005,
                stage2_seed_cc_weight=100.0,
                stage2_seed_cc_threshold=0.05,
                stage2_seed_curvature_weight=0.0001,
                stage2_seed_curvature_threshold=40.0,
                stage2_seed_banana_surf_radius=0.22,
                stage2_seed_tf_current_A=-8.0e4,
                stage2_seed_order=2,
                stage2_seed_banana_init_current_A=-1.0e4,
            )

            self.assertEqual(module.build_stage2_bs_path(args), str(expected_path))

    def test_build_stage2_bs_path_falls_back_to_legacy_basin_hop_without_tf_segment(
        self,
    ):
        module = load_single_stage_example_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "outputs-demo.nc"
            legacy_dir = (
                outputs_dir
                / "R0=0.976-s=0.24-LW=0.0005-CCW=100-CW=0.0001-SR=0.220-Order=2-BH=3-BS=0.01-BSeed=7"
            )
            legacy_dir.mkdir(parents=True)
            expected_path = legacy_dir / "biot_savart_opt.json"
            expected_path.write_text("{}", encoding="utf-8")

            args = SimpleNamespace(
                stage2_bs_path=None,
                stage2_source="local",
                local_stage2_root=tmpdir,
                database_stage2_root="/unused",
                plasma_surf_filename="demo.nc",
                stage2_seed_major_radius=0.976,
                stage2_seed_toroidal_flux=0.24,
                stage2_seed_length_weight=0.0005,
                stage2_seed_cc_weight=100.0,
                stage2_seed_cc_threshold=0.05,
                stage2_seed_curvature_weight=0.0001,
                stage2_seed_curvature_threshold=40.0,
                stage2_seed_banana_surf_radius=0.22,
                stage2_seed_tf_current_A=-8.0e4,
                stage2_seed_order=2,
                stage2_seed_banana_init_current_A=-1.0e4,
            )

            self.assertEqual(module.build_stage2_bs_path(args), str(expected_path))

    def test_build_stage2_bs_path_falls_back_to_legacy_radius_for_local_lookup(self):
        module = load_single_stage_example_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "outputs-demo.nc"
            legacy_dir = (
                outputs_dir
                / "R0=0.976-s=0.24-LW=0.0005-CCW=100-CCT=0.05-CW=0.0001-CT=100-SR=0.220-INITC=-10000-MAXC=16000-TFC=-80000-Order=2-CM=penalty"
            )
            legacy_dir.mkdir(parents=True)
            expected_path = legacy_dir / "biot_savart_opt.json"
            expected_path.write_text("{}", encoding="utf-8")

            args = SimpleNamespace(
                stage2_bs_path=None,
                stage2_source="local",
                local_stage2_root=tmpdir,
                database_stage2_root="/unused",
                plasma_surf_filename="demo.nc",
                stage2_seed_major_radius=0.976,
                stage2_seed_toroidal_flux=0.24,
                stage2_seed_length_weight=0.0005,
                stage2_seed_cc_weight=100.0,
                stage2_seed_cc_threshold=0.05,
                stage2_seed_curvature_weight=0.0001,
                stage2_seed_curvature_threshold=100.0,
                stage2_seed_banana_surf_radius=module.BANANA_WINDING_MINOR_RADIUS_M,
                stage2_seed_tf_current_A=-8.0e4,
                stage2_seed_order=2,
                stage2_seed_banana_init_current_A=-1.0e4,
            )

            self.assertEqual(module.build_stage2_bs_path(args), str(expected_path))

    def test_build_stage2_bs_path_falls_back_to_legacy_radius_for_database_lookup(self):
        module = load_single_stage_example_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "outputs-demo.nc"
            legacy_dir = (
                outputs_dir
                / "MR=0.976-TF=0.24-LW=0.0005-CCW=100-CW=0.0001-SR=0.22-INITC=-10000-TFC=-80000-Order=2"
            )
            legacy_dir.mkdir(parents=True)
            expected_path = legacy_dir / "biot_savart_opt.json"
            expected_path.write_text("{}", encoding="utf-8")

            args = SimpleNamespace(
                stage2_bs_path=None,
                stage2_source="database",
                local_stage2_root="/unused",
                database_stage2_root=tmpdir,
                plasma_surf_filename="demo.nc",
                stage2_seed_major_radius=0.976,
                stage2_seed_toroidal_flux=0.24,
                stage2_seed_length_weight=0.0005,
                stage2_seed_cc_weight=100.0,
                stage2_seed_cc_threshold=0.05,
                stage2_seed_curvature_weight=0.0001,
                stage2_seed_curvature_threshold=100.0,
                stage2_seed_banana_surf_radius=module.BANANA_WINDING_MINOR_RADIUS_M,
                stage2_seed_tf_current_A=-8.0e4,
                stage2_seed_order=2,
                stage2_seed_banana_init_current_A=-1.0e4,
            )

            self.assertEqual(module.build_stage2_bs_path(args), str(expected_path))

    def test_build_stage2_bs_path_discovers_unique_wataru_local_output(self):
        module = load_single_stage_example_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "outputs-demo.nc"
            wataru_dir = (
                outputs_dir
                / "R0=0.976-s=0.24-LW=0.0005-CCW=100-CCT=0.05-CW=0.0001-CT=40-SR=0.220-INITC=-10000-MAXC=16000-TFC=-80000-Order=2-FCM=wataru_proxy_field-PPC=9000-VFC=500-VFT=wataru_vf_template-CM=penalty"
            )
            wataru_dir.mkdir(parents=True)
            expected_path = wataru_dir / "biot_savart_opt.json"
            expected_path.write_text("{}", encoding="utf-8")

            args = SimpleNamespace(
                stage2_bs_path=None,
                stage2_source="local",
                local_stage2_root=tmpdir,
                database_stage2_root="/unused",
                plasma_surf_filename="demo.nc",
                stage2_seed_major_radius=0.976,
                stage2_seed_toroidal_flux=0.24,
                stage2_seed_length_weight=0.0005,
                stage2_seed_cc_weight=100.0,
                stage2_seed_cc_threshold=0.05,
                stage2_seed_curvature_weight=0.0001,
                stage2_seed_curvature_threshold=40.0,
                stage2_seed_banana_surf_radius=0.22,
                stage2_seed_tf_current_A=-8.0e4,
                stage2_seed_order=2,
                stage2_seed_banana_init_current_A=-1.0e4,
            )

            self.assertEqual(module.build_stage2_bs_path(args), str(expected_path))

    def test_single_stage_parse_args_accepts_hardware_search_flags(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--hardware-search-mode",
                "adaptive",
                "--hardware-search-soft-iterations",
                "3",
                "--curvature-traversal-band",
                "0.05",
                "--curvature-traversal-eval-budget",
                "2",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.hardware_search_mode, "adaptive")
        self.assertEqual(args.hardware_search_soft_iterations, 3)
        self.assertEqual(args.curvature_traversal_band, 0.05)
        self.assertEqual(args.curvature_traversal_eval_budget, 2)

    def test_single_stage_parse_args_accepts_shape_weight_flags(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--single-stage-poloidal-weight",
                "4.0",
                "--single-stage-width-weight",
                "2.5",
                "--single-stage-selfint-weight",
                "3.5",
                "--single-stage-available-envelope-reward-weight",
                "1.25",
                "--winding-surface-free-r0",
                "--winding-surface-free-minor",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.single_stage_poloidal_weight, 4.0)
        self.assertEqual(args.single_stage_width_weight, 2.5)
        self.assertEqual(args.single_stage_selfint_weight, 3.5)
        self.assertEqual(args.single_stage_available_envelope_reward_weight, 1.25)
        self.assertTrue(args.winding_surface_free_r0)
        self.assertTrue(args.winding_surface_free_minor)

    def test_single_stage_parse_args_accepts_available_envelope_reward_flag(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--single-stage-available-envelope-reward-weight",
                "12.5",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.single_stage_available_envelope_reward_weight, 12.5)

    def test_single_stage_parse_args_accepts_confinement_objective_flags(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--single-stage-rational-iota-avoidance-weight",
                "1.5",
                "--single-stage-rational-iota-avoidance-max-denominator",
                "13",
                "--single-stage-rational-iota-avoidance-sigma",
                "0.02",
                "--single-stage-iota-noble-pull-weight",
                "2.5",
                "--single-stage-iota-noble-pull-lo",
                "0.276",
                "--single-stage-iota-noble-pull-hi",
                "0.281",
                "--single-stage-iota-pin-weight",
                "1.25",
                "--single-stage-iota-pin-target",
                "0.086",
                "--single-stage-iota-pin-window",
                "0.002",
                "--single-stage-geodesic-curvature-weight",
                "1.75",
                "--single-stage-geodesic-curvature-threshold",
                "31.0",
                "--single-stage-geodesic-curvature-p",
                "4",
                "--single-stage-qa-residual-weight",
                "3.5",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.single_stage_rational_iota_avoidance_weight, 1.5)
        self.assertEqual(
            args.single_stage_rational_iota_avoidance_max_denominator,
            13,
        )
        self.assertEqual(args.single_stage_rational_iota_avoidance_sigma, 0.02)
        self.assertEqual(args.single_stage_iota_noble_pull_weight, 2.5)
        self.assertEqual(args.single_stage_iota_noble_pull_lo, 0.276)
        self.assertEqual(args.single_stage_iota_noble_pull_hi, 0.281)
        self.assertEqual(args.single_stage_iota_pin_weight, 1.25)
        self.assertEqual(args.single_stage_iota_pin_target, 0.086)
        self.assertEqual(args.single_stage_iota_pin_window, 0.002)
        self.assertEqual(args.single_stage_geodesic_curvature_weight, 1.75)
        self.assertEqual(args.single_stage_geodesic_curvature_threshold, 31.0)
        self.assertEqual(args.single_stage_geodesic_curvature_p, 4)
        self.assertEqual(args.single_stage_qa_residual_weight, 3.5)

    def test_single_stage_parse_args_accepts_hardware_sdf_reward_with_sdf_backend(
        self,
    ):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--hardware-keepout-backend",
                "sdf",
                "--hardware-keepout-sdf-manifest",
                "/tmp/hardware_sdf.json",
                "--single-stage-hardware-sdf-free-space-reward-weight",
                "12.5",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(
            args.single_stage_hardware_sdf_free_space_reward_weight,
            12.5,
        )

    def test_single_stage_parse_args_accepts_clearance_hinge_profile_and_softmin(
        self,
    ):
        module = load_single_stage_example_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "clearance_profile.json"
            profile_path.write_text("[0.003, 0.004, 0.005]", encoding="utf-8")
            with patch.object(
                sys,
                "argv",
                [
                    "single_stage_banana_example.py",
                    "--hardware-keepout-backend",
                    "sdf",
                    "--hardware-keepout-sdf-manifest",
                    "/tmp/hardware_sdf.json",
                    "--clearance-hinge-weight",
                    "2.0",
                    "--clearance-hinge-margin-profile-json",
                    str(profile_path),
                    "--clearance-hinge-soft-min-temperature-m",
                    "0.001",
                ],
            ):
                args = module.parse_args()

        self.assertEqual(args.clearance_hinge_margin_profile_json, str(profile_path))
        self.assertTrue(
            np.allclose(
                args.clearance_hinge_target_margin_m,
                np.array([0.003, 0.004, 0.005]),
            )
        )
        self.assertEqual(args.clearance_hinge_soft_min_temperature_m, 0.001)

    def test_single_stage_parse_args_rejects_invalid_clearance_hinge_profile(
        self,
    ):
        module = load_single_stage_example_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "clearance_profile.json"
            profile_path.write_text("[0.003, -0.004]", encoding="utf-8")
            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "single_stage_banana_example.py",
                        "--clearance-hinge-margin-profile-json",
                        str(profile_path),
                    ],
                ),
                self.assertRaises(SystemExit),
            ):
                module.parse_args()

    def test_clearance_hinge_target_margin_result_fields(self):
        module = load_single_stage_example_module()

        fields = module.clearance_hinge_target_margin_result_fields(
            np.array([0.003, 0.005]),
            profile_json="profile.json",
            soft_min_temperature_m=0.001,
        )

        self.assertEqual(
            fields["SINGLE_STAGE_CLEARANCE_HINGE_TARGET_MARGIN_KIND"],
            "profile",
        )
        self.assertEqual(
            fields["SINGLE_STAGE_CLEARANCE_HINGE_TARGET_MARGIN_COUNT"],
            2,
        )
        self.assertEqual(
            fields["SINGLE_STAGE_CLEARANCE_HINGE_TARGET_MARGIN_MIN_M"],
            0.003,
        )
        self.assertEqual(
            fields["SINGLE_STAGE_CLEARANCE_HINGE_TARGET_MARGIN_MAX_M"],
            0.005,
        )
        self.assertEqual(
            fields["SINGLE_STAGE_CLEARANCE_HINGE_SOFT_MIN_TEMPERATURE_M"],
            0.001,
        )
        self.assertEqual(
            fields["SINGLE_STAGE_CLEARANCE_HINGE_REDUCTION"],
            "soft_min",
        )

    def test_single_stage_parse_args_rejects_negative_available_envelope_reward_weight(
        self,
    ):
        module = load_single_stage_example_module()

        with (
            patch.object(
                sys,
                "argv",
                [
                    "single_stage_banana_example.py",
                    "--single-stage-available-envelope-reward-weight",
                    "-0.1",
                ],
            ),
            self.assertRaises(SystemExit),
        ):
            module.parse_args()

    def test_single_stage_parse_args_rejects_invalid_confinement_objective_flags(
        self,
    ):
        module = load_single_stage_example_module()

        cases = [
            ("--single-stage-rational-iota-avoidance-weight", "-0.1"),
            ("--single-stage-rational-iota-avoidance-weight", "nan"),
            ("--single-stage-rational-iota-avoidance-max-denominator", "0"),
            ("--single-stage-rational-iota-avoidance-sigma", "0.0"),
            ("--single-stage-rational-iota-avoidance-sigma", "inf"),
            ("--single-stage-iota-noble-pull-weight", "-0.1"),
            ("--single-stage-iota-noble-pull-weight", "nan"),
            ("--single-stage-iota-pin-weight", "-0.1"),
            ("--single-stage-iota-pin-weight", "nan"),
            ("--single-stage-iota-pin-target", "inf"),
            ("--single-stage-iota-pin-window", "0.0"),
            ("--single-stage-iota-pin-window", "nan"),
            ("--single-stage-geodesic-curvature-weight", "-0.1"),
            ("--single-stage-geodesic-curvature-weight", "nan"),
            ("--single-stage-geodesic-curvature-threshold", "0.0"),
            ("--single-stage-geodesic-curvature-threshold", "inf"),
            ("--single-stage-geodesic-curvature-p", "0"),
            ("--single-stage-qa-residual-weight", "-0.1"),
            ("--single-stage-qa-residual-weight", "inf"),
        ]
        for flag, value in cases:
            with self.subTest(flag=flag):
                with (
                    patch.object(
                        sys,
                        "argv",
                        [
                            "single_stage_banana_example.py",
                            flag,
                            value,
                        ],
                    ),
                    self.assertRaises(SystemExit),
                ):
                    module.parse_args()

    def test_single_stage_parse_args_rejects_invalid_noble_pull_window(self):
        module = load_single_stage_example_module()

        cases = [
            ("0.3", "0.2"),
            ("nan", "0.281"),
            ("0.276", "inf"),
        ]
        for lo, hi in cases:
            with self.subTest(lo=lo, hi=hi):
                with (
                    patch.object(
                        sys,
                        "argv",
                        [
                            "single_stage_banana_example.py",
                            "--single-stage-iota-noble-pull-lo",
                            lo,
                            "--single-stage-iota-noble-pull-hi",
                            hi,
                        ],
                    ),
                    self.assertRaises(SystemExit),
                ):
                    module.parse_args()

    def test_single_stage_parse_args_rejects_hardware_sdf_reward_without_sdf_backend(
        self,
    ):
        module = load_single_stage_example_module()

        with (
            patch.object(
                sys,
                "argv",
                [
                    "single_stage_banana_example.py",
                    "--single-stage-hardware-sdf-free-space-reward-weight",
                    "0.5",
                ],
            ),
            self.assertRaises(SystemExit),
        ):
            module.parse_args()

    def test_single_stage_parse_args_rejects_hardware_sdf_reward_without_manifest(
        self,
    ):
        module = load_single_stage_example_module()

        with (
            patch.object(
                sys,
                "argv",
                [
                    "single_stage_banana_example.py",
                    "--hardware-keepout-backend",
                    "sdf",
                    "--single-stage-hardware-sdf-free-space-reward-weight",
                    "0.5",
                ],
            ),
            self.assertRaises(SystemExit),
        ):
            module.parse_args()

    def test_single_stage_parse_args_accepts_winding_surface_size_flags(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--winding-surface-free-r0",
                "--winding-surface-free-minor",
            ],
        ):
            args = module.parse_args()

        self.assertTrue(args.winding_surface_free_r0)
        self.assertTrue(args.winding_surface_free_minor)

    def test_single_stage_parse_args_accepts_frontier_ceiling_flags(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--frontier-iota-ceiling",
                "0.30",
                "--frontier-volume-ceiling",
                "0.18",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.frontier_iota_ceiling, 0.30)
        self.assertEqual(args.frontier_volume_ceiling, 0.18)

    def test_single_stage_parse_args_frontier_ceiling_defaults_none(self):
        module = load_single_stage_example_module()

        with (
            patch.object(sys, "argv", ["single_stage_banana_example.py"]),
            patch.dict(os.environ, {}, clear=True),
        ):
            args = module.parse_args()

        self.assertIsNone(args.frontier_iota_ceiling)
        self.assertIsNone(args.frontier_volume_ceiling)

    def test_single_stage_parse_args_frontier_ceiling_env_backed(self):
        module = load_single_stage_example_module()

        with (
            patch.object(sys, "argv", ["single_stage_banana_example.py"]),
            patch.dict(
                os.environ,
                {
                    "FRONTIER_IOTA_CEILING": "0.27",
                    "FRONTIER_VOLUME_CEILING": "0.16",
                },
                clear=True,
            ),
        ):
            args = module.parse_args()

        self.assertEqual(args.frontier_iota_ceiling, 0.27)
        self.assertEqual(args.frontier_volume_ceiling, 0.16)

    def test_single_stage_parse_args_defaults_engineering_contract_terms_on(self):
        module = load_single_stage_example_module()

        with (
            patch.object(sys, "argv", ["single_stage_banana_example.py"]),
            patch.dict(os.environ, {}, clear=True),
        ):
            args = module.parse_args()

        self.assertEqual(
            args.single_stage_hardware_keepout_weight,
            module.SINGLE_STAGE_HARDWARE_KEEPOUT_WEIGHT_DEFAULT,
        )
        self.assertEqual(
            args.single_stage_vessel_keepout_weight,
            module.SINGLE_STAGE_VESSEL_KEEPOUT_WEIGHT_DEFAULT,
        )
        self.assertEqual(
            args.coil_force_weight,
            module.SINGLE_STAGE_COIL_FORCE_WEIGHT_DEFAULT,
        )
        self.assertEqual(
            args.coil_force_conductor_radius,
            module.SINGLE_STAGE_COIL_FORCE_CONDUCTOR_RADIUS_DEFAULT_M,
        )
        self.assertEqual(
            args.hardware_keepout_json,
            module.DEFAULT_HARDWARE_KEEPOUT_JSON_PATH,
        )
        self.assertEqual(
            args.hardware_keepout_glb,
            module.DEFAULT_HARDWARE_KEEPOUT_GLB_PATH,
        )
        self.assertGreater(args.single_stage_hardware_keepout_weight, 0.0)
        self.assertGreater(args.single_stage_vessel_keepout_weight, 0.0)
        self.assertEqual(args.single_stage_available_envelope_reward_weight, 0.0)
        self.assertFalse(args.winding_surface_free_r0)
        self.assertFalse(args.winding_surface_free_minor)
        self.assertGreater(args.coil_force_weight, 0.0)
        self.assertGreater(args.coil_force_conductor_radius, 0.0)

    def test_single_stage_parse_args_accepts_hardware_keepout_glb(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--hardware-keepout-glb",
                "/tmp/hbt_assembly.glb",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.hardware_keepout_glb, "/tmp/hbt_assembly.glb")

    def test_single_stage_parse_args_accepts_residue_objective_flags(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--residue-objective-weight",
                "0.25",
                "--residue-objective-targets-json",
                "targets.json",
                "--residue-objective-seeds-json",
                "seeds.json",
                "--residue-objective-selected-chain-config",
                "one-over-11-12",
                "--residue-objective-axis-r",
                "1.02",
                "--residue-objective-det-tolerance",
                "2e-5",
                "--residue-promotion-gate",
                "--residue-promotion-gate-threshold",
                "0.2",
                "--residue-promotion-gate-samples",
                "3",
                "--residue-promotion-gate-radius",
                "1e-4",
                "--residue-promotion-gate-seed",
                "17",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.residue_objective_weight, 0.25)
        self.assertEqual(args.residue_objective_targets_json, "targets.json")
        self.assertEqual(args.residue_objective_seeds_json, "seeds.json")
        self.assertEqual(
            args.residue_objective_selected_chain_config, "one-over-11-12"
        )
        self.assertEqual(args.residue_objective_axis_r, 1.02)
        self.assertEqual(args.residue_objective_det_tolerance, 2.0e-5)
        self.assertTrue(args.residue_promotion_gate)
        self.assertEqual(args.residue_promotion_gate_threshold, 0.2)
        self.assertEqual(args.residue_promotion_gate_samples, 3)
        self.assertEqual(args.residue_promotion_gate_radius, 1.0e-4)
        self.assertEqual(args.residue_promotion_gate_seed, 17)
        self.assertEqual(
            args.residue_objective_samples_per_full_torus,
            module.DEFAULT_RESIDUE_OBJECTIVE_SAMPLES_PER_FULL_TORUS,
        )

    def test_selected_residue_chain_requires_exact_target_pairs(self):
        module = load_single_stage_example_module()
        selected_targets = [
            SimpleNamespace(p=1, q=11),
            SimpleNamespace(p=1, q=12),
        ]

        module.validate_residue_selected_chain_targets(
            selected_targets, "one-over-11-12"
        )

        with self.assertRaisesRegex(ValueError, "requires exactly"):
            module.validate_residue_selected_chain_targets(
                [SimpleNamespace(p=1, q=11), SimpleNamespace(p=1, q=10)],
                "one-over-11-12",
            )
        with self.assertRaisesRegex(ValueError, "requires exactly"):
            module.validate_residue_selected_chain_targets(
                [SimpleNamespace(p=1, q=11), SimpleNamespace(p=1, q=11)],
                "one-over-11-12",
            )

    def test_residue_promotion_gate_report_restores_biot_savart_dofs(self):
        module = load_single_stage_example_module()

        class FakeResidueObjective:
            def __init__(self):
                self.biot_savart = SimpleNamespace(x=np.array([1.0, 2.0, 3.0]))
                self.recompute_count = 0

            def recompute_bell(self, parent=None):
                self.recompute_count += 1

            def to_json_dict(self):
                return {
                    "branches": [
                        {
                            "target_id": "p1_q11",
                            "branch": "O",
                            "status": "converged",
                            "residue": 0.1,
                        },
                        {
                            "target_id": "p1_q12",
                            "branch": "X",
                            "status": "converged",
                            "residue": -0.2,
                        },
                    ]
                }

        args = SimpleNamespace(
            residue_promotion_gate=True,
            residue_objective_selected_chain_config="one-over-11-12",
            residue_promotion_gate_threshold=0.25,
            residue_promotion_gate_samples=2,
            residue_promotion_gate_radius=1.0e-4,
            residue_promotion_gate_seed=7,
        )
        residue_objective = FakeResidueObjective()
        original_x = residue_objective.biot_savart.x.copy()

        report = module.residue_promotion_gate_report(args, residue_objective)

        self.assertTrue(report["enabled"])
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["sample_reports"]), 3)
        np.testing.assert_allclose(residue_objective.biot_savart.x, original_x)

    def test_single_stage_parse_args_accepts_magnetic_well_flags(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--magnetic-well-weight",
                "5.0",
                "--magnetic-well-target",
                "-0.05",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.magnetic_well_weight, 5.0)
        self.assertEqual(args.magnetic_well_target, -0.05)

    def test_single_stage_parse_args_edge_iota_lane_sets_defaults(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--single-stage-lane",
                "edge_delivered_iota_lane",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.single_stage_lane, "edge_delivered_iota_lane")
        self.assertEqual(args.surface_mode, module.PUBLISHED_MULTISURFACE)
        self.assertEqual(
            args.published_surface_preset,
            module.PUBLISHED_PRESET_EDGE_DELIVERED_IOTA_LANE,
        )
        self.assertEqual(args.iota_profile_weight, 1.0)
        self.assertEqual(args.iota_profile_surface_weights, "0.25,0.5,1,4")
        self.assertEqual(args.topology_seed_mode, module.SEED_MODE_EXTENDED_SURFACE)
        self.assertEqual(args.winding_surface_free_mpol, 1)
        self.assertEqual(args.winding_surface_free_ntor, 1)
        self.assertTrue(args.winding_surface_free_r0)
        self.assertTrue(args.winding_surface_free_minor)

    def test_edge_iota_lane_weights_match_explicit_surface_fractions(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--single-stage-lane",
                "edge_delivered_iota_lane",
                "--published-surface-fractions",
                "0.75,1.0",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.published_surface_fractions, "0.75,1.0")
        self.assertEqual(args.iota_profile_surface_weights, "0.25,4")
        self.assertEqual(
            module.parse_single_stage_iota_profile_surface_weights(
                args.iota_profile_surface_weights
            ),
            (0.25, 4.0),
        )

    def test_single_stage_parse_args_uses_measured_lbfgsb_maxcor_default(self):
        module = load_single_stage_example_module()

        with patch.object(sys, "argv", ["single_stage_banana_example.py"]):
            args = module.parse_args()

        self.assertEqual(args.maxcor, module.DEFAULT_LBFGSB_MAXCOR)
        self.assertEqual(args.maxcor, 40)

    def test_single_stage_parse_args_help_renders_with_fd_flag_text(self):
        module = load_single_stage_example_module()

        with (
            patch.object(
                sys,
                "argv",
                [
                    "single_stage_banana_example.py",
                    "--help",
                ],
            ),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            with self.assertRaises(SystemExit) as excinfo:
                module.parse_args()

        help_text = stdout.getvalue()
        self.assertEqual(excinfo.exception.code, 0)
        self.assertIn("--banana-current-fd-diagnostics", help_text)
        self.assertIn("+/-1% perturbations", help_text)
        self.assertIn("direct_proxy_consistency_validations", help_text)
        self.assertIn("real_field_nonzero_winding_validations", help_text)

    def test_single_stage_parse_args_accepts_goal_mode_flag(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--single-stage-goal-mode",
                "frontier",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.single_stage_goal_mode, "frontier")

    def test_single_stage_parse_args_accepts_seed_regime_flag(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--seed-regime",
                "bridge_only",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.seed_regime, "bridge_only")

    def test_single_stage_parse_args_accepts_tf_current_A(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--tf-current-A",
                "-80000",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.tf_current_A, -80000.0)

    def test_single_stage_parse_args_accepts_resume_replay_flags(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--single-stage-resume-bs-path",
                "archives/biot_savart_opt.json",
                "--offspec-replay-debug-only",
                "--accept-offspec-r0-seed",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(
            args.single_stage_resume_bs_path,
            "archives/biot_savart_opt.json",
        )
        self.assertTrue(args.offspec_replay_debug_only)
        self.assertTrue(args.accept_offspec_r0_seed)

    def test_single_stage_parse_args_accepts_offspec_coil_length_flag(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--length-target",
                "3.0",
                "--accept-offspec-coil-length",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.length_target, 3.0)
        self.assertTrue(args.accept_offspec_coil_length)
        self.assertEqual(
            module.validate_coil_length_target(
                args.length_target,
                accept_offspec_coil_length=args.accept_offspec_coil_length,
                field_name="--length-target",
            ),
            3.0,
        )

    def test_single_stage_resume_seed_requires_debug_only_role(self):
        module = load_single_stage_example_module()

        args = SimpleNamespace(
            single_stage_resume_bs_path="archives/biot_savart_opt.json",
            stage2_bs_path=None,
            offspec_replay_debug_only=False,
            strict_vacuum_current=False,
        )

        with self.assertRaisesRegex(
            ValueError,
            "--single-stage-resume-bs-path requires "
            "--offspec-replay-debug-only or --strict-vacuum-current",
        ):
            module.resolve_single_stage_seed_artifact(args)

    def test_single_stage_resume_seed_allows_strict_vacuum_role(self):
        module = load_single_stage_example_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            seed_path = Path(tmpdir) / "biot_savart_opt.json"
            seed_path.write_text("{}", encoding="utf-8")
            results_path = Path(tmpdir) / "results.json"
            results_path.write_text(
                json.dumps(
                    {
                        "MAJOR_RADIUS": 0.976,
                        "TOROIDAL_FLUX": 0.24,
                        "banana_surf_radius": 0.21,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                single_stage_resume_bs_path=str(seed_path),
                stage2_bs_path=None,
                offspec_replay_debug_only=False,
                strict_vacuum_current=True,
            )

            resolved = module.resolve_single_stage_seed_artifact(args)

        self.assertEqual(resolved[0], str(seed_path))
        self.assertEqual(resolved[1], results_path)
        self.assertEqual(resolved[3], module.SEED_ARTIFACT_ROLE_SINGLE_STAGE_RESUME)

    def test_single_stage_parse_args_accepts_strict_vacuum_current(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--single-stage-resume-bs-path",
                "archives/biot_savart_opt.json",
                "--strict-vacuum-current",
                "--strict-vacuum-lineage=recent_stage1_candidate",
                "--stage1-candidate-id=s01_3240f0",
            ],
        ):
            args = module.parse_args()

        self.assertTrue(args.strict_vacuum_current)
        self.assertEqual(args.strict_vacuum_lineage, "recent_stage1_candidate")
        self.assertEqual(args.stage1_candidate_id, "s01_3240f0")
        self.assertEqual(
            args.single_stage_resume_bs_path,
            "archives/biot_savart_opt.json",
        )

    def test_strict_vacuum_current_args_require_lineage(self):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            strict_vacuum_current=True,
            strict_vacuum_lineage=None,
            stage1_candidate_id=None,
            offspec_replay_debug_only=False,
            boozer_I=None,
            plasma_current_A=None,
            finite_current_mode=None,
        )

        with self.assertRaisesRegex(ValueError, "--strict-vacuum-lineage"):
            module.validate_strict_vacuum_current_args(
                args,
                ["--strict-vacuum-current"],
            )

    def test_strict_vacuum_current_args_require_stage1_candidate_id(self):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            strict_vacuum_current=True,
            strict_vacuum_lineage="recent_stage1_candidate",
            stage1_candidate_id=None,
            offspec_replay_debug_only=False,
            boozer_I=None,
            plasma_current_A=None,
            finite_current_mode=None,
        )

        with self.assertRaisesRegex(ValueError, "--stage1-candidate-id"):
            module.validate_strict_vacuum_current_args(
                args,
                [
                    "--strict-vacuum-current",
                    "--strict-vacuum-lineage=recent_stage1_candidate",
                ],
            )

    def test_strict_vacuum_current_args_reject_current_flags(self):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            strict_vacuum_current=True,
            strict_vacuum_lineage="recent_stage1_candidate",
            stage1_candidate_id="s01_3240f0",
            offspec_replay_debug_only=False,
            boozer_I=None,
            plasma_current_A=None,
            finite_current_mode=None,
        )

        with self.assertRaisesRegex(ValueError, "forbidden_flags"):
            module.validate_strict_vacuum_current_args(
                args,
                ["--strict-vacuum-current", "--plasma-current-A=1.0"],
            )

    def test_strict_vacuum_current_args_reject_env_finite_mode(self):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            strict_vacuum_current=True,
            strict_vacuum_lineage="legacy_control",
            stage1_candidate_id=None,
            offspec_replay_debug_only=False,
            boozer_I=None,
            plasma_current_A=None,
            finite_current_mode="wataru_proxy_field",
        )

        with self.assertRaisesRegex(ValueError, "--finite-current-mode"):
            module.validate_strict_vacuum_current_args(
                args,
                ["--strict-vacuum-current"],
            )

    def test_project_strict_vacuum_seed_biot_savart_drops_proxy_vf_groups(self):
        module = load_single_stage_example_module()

        @dataclass(frozen=True)
        class _Partitions:
            tf_coils: tuple
            banana_coils: tuple
            proxy_coils: tuple
            vf_coils: tuple
            num_tf_coils: int
            num_banana_coils: int
            num_proxy_coils: int
            num_vf_coils: int

        tf_coils = (object(), object())
        banana_coils = (object(),)
        proxy_coils = (object(),)
        vf_coils = (object(), object())
        source_biot_savart = SimpleNamespace(
            coils=[*tf_coils, *banana_coils, *proxy_coils, *vf_coils]
        )
        source_partitions = _Partitions(
            tf_coils=tf_coils,
            banana_coils=banana_coils,
            proxy_coils=proxy_coils,
            vf_coils=vf_coils,
            num_tf_coils=len(tf_coils),
            num_banana_coils=len(banana_coils),
            num_proxy_coils=len(proxy_coils),
            num_vf_coils=len(vf_coils),
        )

        with patch.object(
            module,
            "BiotSavart",
            side_effect=lambda coils: SimpleNamespace(coils=coils),
        ):
            projected_biot_savart, projected_partitions = (
                module.project_strict_vacuum_seed_biot_savart(
                    source_biot_savart,
                    source_partitions,
                )
            )

        self.assertEqual(projected_biot_savart.coils, [*tf_coils, *banana_coils])
        self.assertEqual(projected_partitions.proxy_coils, ())
        self.assertEqual(projected_partitions.vf_coils, ())
        self.assertEqual(projected_partitions.num_proxy_coils, 0)
        self.assertEqual(projected_partitions.num_vf_coils, 0)

    def test_strict_vacuum_current_settings_override_stage2_proxy_mode(self):
        module = load_single_stage_example_module()

        settings = module.apply_strict_vacuum_current_settings(
            SimpleNamespace(strict_vacuum_current=True),
            {
                "boozer_I": 0.001,
                "plasma_current_A": 1500.0,
                "input_source": "finite_current_mode_default",
                "boozer_current_convention": "wataru",
                "mode": "wataru_proxy_field",
                "effective_mode": "finite_current",
            },
        )

        self.assertEqual(settings["boozer_I"], 0.0)
        self.assertEqual(settings["plasma_current_A"], 0.0)
        self.assertEqual(settings["input_source"], "strict_vacuum_current")
        self.assertIsNone(settings["boozer_current_convention"])
        self.assertIsNone(settings["mode"])
        self.assertEqual(settings["effective_mode"], "vacuum")

    def test_single_stage_parse_args_accepts_stage2_seed_surf_path(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--stage2-seed-surf-path",
                "archives/surf_opt_boozer_surface.json",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(
            args.stage2_seed_surf_path,
            "archives/surf_opt_boozer_surface.json",
        )

    def test_single_stage_parse_args_accepts_warm_start_surface_stem(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--warm-start-surface-stem",
                "recovery/surf_best_feasible",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.warm_start_surface_stem, "recovery/surf_best_feasible")

    def test_single_stage_parse_args_accepts_warm_continue_step_norm_limit(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--warm-continue-step-norm-limit",
                "0.025",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.warm_continue_step_norm_limit, 0.025)

    def test_single_stage_parse_args_accepts_frontier_volume_weight_flag(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--frontier-volume-weight",
                "200",
            ],
        ):
            args = module.parse_args()

        self.assertAlmostEqual(args.frontier_volume_weight, 200.0)

    def test_single_stage_parse_args_accepts_frontier_scalarization_flags(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--frontier-scalarization-type",
                "achievement_chebyshev_sweep_v1",
                "--frontier-chebyshev-rho",
                "0.02",
                "--frontier-chebyshev-weight-iota",
                "2.0",
                "--epsilon-constraint-qa-max",
                "0.011",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(
            args.frontier_scalarization_type,
            "achievement_chebyshev_sweep_v1",
        )
        self.assertAlmostEqual(args.frontier_chebyshev_rho, 0.02)
        self.assertAlmostEqual(args.frontier_chebyshev_weight_iota, 2.0)
        self.assertAlmostEqual(args.epsilon_constraint_qa_max, 0.011)

    def test_single_stage_parse_args_reads_goal_mode_from_environment(self):
        module = load_single_stage_example_module()

        with (
            patch.dict(os.environ, {"SINGLE_STAGE_GOAL_MODE": "frontier"}, clear=False),
            patch.object(
                sys,
                "argv",
                ["single_stage_banana_example.py"],
            ),
        ):
            args = module.parse_args()

        self.assertEqual(args.single_stage_goal_mode, "frontier")

    def test_single_stage_parse_args_defaults_target_mode_to_historical_iota(self):
        module = load_single_stage_example_module()

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(sys, "argv", ["single_stage_banana_example.py"]),
        ):
            args = module.parse_args()

        self.assertEqual(args.single_stage_goal_mode, "target")
        self.assertAlmostEqual(
            args.iota_target,
            module.DEFAULT_SINGLE_STAGE_IOTA_TARGET,
        )

    def test_single_stage_parse_args_defaults_frontier_mode_to_off_resonance_iota(
        self,
    ):
        module = load_single_stage_example_module()

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                sys,
                "argv",
                ["single_stage_banana_example.py", "--single-stage-goal-mode", "frontier"],
            ),
        ):
            args = module.parse_args()

        self.assertEqual(args.single_stage_goal_mode, "frontier")
        self.assertAlmostEqual(
            args.iota_target,
            module.DEFAULT_FRONTIER_SINGLE_STAGE_IOTA_TARGET,
        )

    def test_single_stage_parse_args_iota_env_overrides_frontier_default(self):
        module = load_single_stage_example_module()

        with (
            patch.dict(os.environ, {"IOTA_TARGET": "0.19"}, clear=True),
            patch.object(
                sys,
                "argv",
                ["single_stage_banana_example.py", "--single-stage-goal-mode", "frontier"],
            ),
        ):
            args = module.parse_args()

        self.assertEqual(args.single_stage_goal_mode, "frontier")
        self.assertAlmostEqual(args.iota_target, 0.19)

    def test_frontier_goal_mode_warning_message_reports_scale_and_unsaturated_reward(
        self,
    ):
        module = load_single_stage_example_module()
        frontier_goal_config = make_frontier_goal_config(
            module,
            iota_reference=0.15,
            qs_reference=2.0e-4,
        )

        warning = module.frontier_goal_mode_warning_message(frontier_goal_config)

        self.assertIn("normalized tradeoff score", warning)
        self.assertIn("iota_ref=0.150000", warning)
        self.assertIn("1.000000e-05", warning)

    def test_apply_frontier_scalarization_override_uses_chebyshev_lane(self):
        module = load_single_stage_example_module()

        class _ScalarObjective:
            def __init__(self, value, grad):
                self._value = value
                self._grad = np.asarray(grad, dtype=float)

            def J(self):
                return self._value

            def dJ(self):
                return self._grad

        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_GOAL_CONFIG = make_frontier_goal_config(
            module,
            scalarization_type="achievement_chebyshev_sweep_v1",
            chebyshev_rho=0.02,
            chebyshev_weight_iota=2.0,
            chebyshev_weight_volume=1.5,
            chebyshev_weight_qa=1.0,
            chebyshev_weight_boozer=0.5,
        )
        module.surface_iota_terms = [_ScalarObjective(0.13, [1.0, 0.0])]
        module.surface_volume_term = _ScalarObjective(0.09, [0.0, 1.0])
        module.EFFECTIVE_RES_WEIGHT = 1.0
        module.EFFECTIVE_IOTAS_WEIGHT = 1.0
        module.EFFECTIVE_VOLUME_WEIGHT = 1.0
        module.LENGTH_WEIGHT = 1.0
        module.CC_WEIGHT = 0.0
        module.CS_WEIGHT = 0.0
        module.CURVATURE_WEIGHT = 0.0

        objective_eval = {
            "total": 0.0,
            "grad": np.zeros(2),
            "J_QS": 1.2e-4,
            "dJ_QS": np.array([0.5, 0.0]),
            "J_QS_objective": 1.2,
            "dJ_QS_objective": np.array([0.5, 0.0]),
            "J_Boozer": 2.0e-6,
            "dJ_Boozer": np.array([0.0, 0.4]),
            "J_Boozer_objective": 2.0,
            "dJ_Boozer_objective": np.array([0.0, 0.4]),
            "J_iota": -0.1,
            "dJ_iota": np.array([-0.3, 0.0]),
            "J_volume": -0.2,
            "dJ_volume": np.array([0.0, -0.2]),
            "J_len": 0.05,
            "dJ_len": np.array([0.1, 0.1]),
            "J_cc": 0.0,
            "dJ_cc": np.zeros(2),
            "J_cs": 0.0,
            "dJ_cs": np.zeros(2),
            "J_curvature": 0.0,
            "dJ_curvature": np.zeros(2),
            "J_hardware_keepout": 0.0,
            "dJ_hardware_keepout": np.zeros(2),
            "J_vessel_keepout": 0.0,
            "dJ_vessel_keepout": np.zeros(2),
        }

        scalarized = module.apply_frontier_scalarization_override(objective_eval)

        self.assertEqual(
            scalarized["frontier_scalarization_type"],
            "achievement_chebyshev_sweep_v1",
        )
        self.assertIn("frontier_chebyshev_deltas", scalarized)
        self.assertNotAlmostEqual(
            scalarized["frontier_goal_total"],
            1.2 + 2.0 - 0.1 - 0.2,
        )

    def test_apply_frontier_scalarization_override_adds_epsilon_search_penalty(self):
        module = load_single_stage_example_module()

        class _ScalarObjective:
            def __init__(self, value, grad):
                self._value = value
                self._grad = np.asarray(grad, dtype=float)

            def J(self):
                return self._value

            def dJ(self):
                return self._grad

        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_GOAL_CONFIG = make_frontier_goal_config(
            module,
            scalarization_type="epsilon_constraint_sweep_v1",
            epsilon_constraint_qa_max=1.0e-4,
            epsilon_constraint_boozer_max=1.0e-6,
        )
        module.surface_iota_terms = [_ScalarObjective(0.13, [1.0, 0.0])]
        module.surface_volume_term = _ScalarObjective(0.09, [0.0, 1.0])
        module.EFFECTIVE_RES_WEIGHT = 1.0
        module.EFFECTIVE_IOTAS_WEIGHT = 1.0
        module.EFFECTIVE_VOLUME_WEIGHT = 1.0
        module.LENGTH_WEIGHT = 1.0
        module.CC_WEIGHT = 0.0
        module.CS_WEIGHT = 0.0
        module.CURVATURE_WEIGHT = 0.0

        objective_eval = {
            "total": 0.0,
            "grad": np.zeros(2),
            "J_QS": 4.0e-4,
            "dJ_QS": np.array([0.5, 0.0]),
            "J_QS_objective": 1.2,
            "dJ_QS_objective": np.array([0.5, 0.0]),
            "J_Boozer": 5.0e-6,
            "dJ_Boozer": np.array([0.0, 0.4]),
            "J_Boozer_objective": 2.0,
            "dJ_Boozer_objective": np.array([0.0, 0.4]),
            "J_iota": -0.1,
            "dJ_iota": np.array([-0.3, 0.0]),
            "J_volume": -0.2,
            "dJ_volume": np.array([0.0, -0.2]),
            "J_len": 0.05,
            "dJ_len": np.array([0.1, 0.1]),
            "J_cc": 0.0,
            "dJ_cc": np.zeros(2),
            "J_cs": 0.0,
            "dJ_cs": np.zeros(2),
            "J_curvature": 0.0,
            "dJ_curvature": np.zeros(2),
            "J_hardware_keepout": 0.0,
            "dJ_hardware_keepout": np.zeros(2),
            "J_vessel_keepout": 0.0,
            "dJ_vessel_keepout": np.zeros(2),
        }

        scalarized = module.apply_frontier_scalarization_override(objective_eval)

        self.assertEqual(
            scalarized["frontier_scalarization_type"],
            "epsilon_constraint_sweep_v1",
        )
        self.assertGreater(scalarized["frontier_epsilon_penalty"], 0.0)
        self.assertIn("qa_error", scalarized["frontier_epsilon_constraints"])

    def test_apply_frontier_scalarization_override_is_noop_outside_frontier_mode(self):
        module = load_single_stage_example_module()
        module.SINGLE_STAGE_GOAL_MODE = "target"
        module.FRONTIER_GOAL_CONFIG = None
        if hasattr(module, "surface_iota_terms"):
            delattr(module, "surface_iota_terms")
        if hasattr(module, "surface_volume_term"):
            delattr(module, "surface_volume_term")

        objective_eval = {
            "total": 1.23,
            "grad": np.array([0.1, -0.2]),
            "J_QS": 1.2e-4,
            "dJ_QS": np.array([0.5, 0.0]),
            "J_Boozer": 2.0e-6,
            "dJ_Boozer": np.array([0.0, 0.4]),
            "J_iota": -0.1,
            "dJ_iota": np.array([-0.3, 0.0]),
        }

        scalarized = module.apply_frontier_scalarization_override(objective_eval)

        self.assertEqual(set(scalarized.keys()), set(objective_eval.keys()))
        self.assertIsNot(scalarized, objective_eval)
        np.testing.assert_allclose(scalarized["grad"], objective_eval["grad"])

    def test_apply_frontier_scalarization_override_projects_metric_gradients(self):
        module = load_single_stage_example_module()
        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_GOAL_CONFIG = make_frontier_goal_config(
            module,
            scalarization_type="achievement_chebyshev_sweep_v1",
            chebyshev_rho=0.02,
        )
        module.JF = object()
        module.surface_iota_terms = [
            FakeProjectedObjective(0.13, [1.0, 0.0], [0.0, 0.0, 1.0, 0.0])
        ]
        module.surface_volume_term = FakeProjectedObjective(
            0.09,
            [0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        )
        module.EFFECTIVE_RES_WEIGHT = 1.0
        module.EFFECTIVE_IOTAS_WEIGHT = 1.0
        module.EFFECTIVE_VOLUME_WEIGHT = 1.0
        module.LENGTH_WEIGHT = 1.0
        module.CC_WEIGHT = 0.0
        module.CS_WEIGHT = 0.0
        module.CURVATURE_WEIGHT = 0.0

        objective_eval = {
            "total": 0.0,
            "grad": np.zeros(4),
            "J_QS": 1.2e-4,
            "dJ_QS": np.array([0.5, 0.0, 0.0, 0.0]),
            "J_QS_objective": 1.2,
            "dJ_QS_objective": np.array([0.5, 0.0, 0.0, 0.0]),
            "J_Boozer": 2.0e-6,
            "dJ_Boozer": np.array([0.0, 0.4, 0.0, 0.0]),
            "J_Boozer_objective": 2.0,
            "dJ_Boozer_objective": np.array([0.0, 0.4, 0.0, 0.0]),
            "J_iota": -0.1,
            "dJ_iota": np.array([-0.3, 0.0, 0.0, 0.0]),
            "J_volume": -0.2,
            "dJ_volume": np.array([0.0, -0.2, 0.0, 0.0]),
            "J_len": 0.05,
            "dJ_len": np.array([0.1, 0.1, 0.0, 0.0]),
            "J_cc": 0.0,
            "dJ_cc": np.zeros(4),
            "J_cs": 0.0,
            "dJ_cs": np.zeros(4),
            "J_curvature": 0.0,
            "dJ_curvature": np.zeros(4),
            "J_hardware_keepout": 0.0,
            "dJ_hardware_keepout": np.zeros(4),
            "J_vessel_keepout": 0.0,
            "dJ_vessel_keepout": np.zeros(4),
        }

        scalarized = module.apply_frontier_scalarization_override(objective_eval)

        self.assertEqual(scalarized["frontier_goal_grad"].shape, (4,))
        np.testing.assert_allclose(
            scalarized["dJ_iota_metric"],
            [0.0, 0.0, 1.0, 0.0],
        )
        np.testing.assert_allclose(
            scalarized["dJ_volume_metric"],
            [0.0, 0.0, 0.0, 1.0],
        )

    def test_apply_frontier_scalarization_override_rebuilds_alm_once(self):
        module = load_single_stage_example_module()

        class _ScalarObjective:
            def __init__(self, value, grad):
                self._value = value
                self._grad = np.asarray(grad, dtype=float)

            def J(self):
                return self._value

            def dJ(self):
                return self._grad

        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_GOAL_CONFIG = make_frontier_goal_config(module)
        module.surface_iota_terms = [_ScalarObjective(0.13, [0.0, 0.0])]
        module.surface_volume_term = _ScalarObjective(0.09, [0.0, 0.0])
        module.EFFECTIVE_RES_WEIGHT = 1.25
        module.EFFECTIVE_IOTAS_WEIGHT = 1.5
        module.EFFECTIVE_VOLUME_WEIGHT = 0.75
        module.LENGTH_WEIGHT = 2.0
        module.CC_WEIGHT = 0.0
        module.CS_WEIGHT = 0.0
        module.CURVATURE_WEIGHT = 0.0
        module.SINGLE_STAGE_POLOIDAL_WEIGHT = 0.0
        module.SINGLE_STAGE_WIDTH_WEIGHT = 0.0
        module.SINGLE_STAGE_SELFINT_WEIGHT = 0.0
        module.SINGLE_STAGE_HARDWARE_KEEPOUT_WEIGHT = 0.0
        module.SINGLE_STAGE_VESSEL_KEEPOUT_WEIGHT = 0.0
        module.MSC_WEIGHT = 0.0
        module.ARCLEN_WEIGHT = 0.0
        module.LINKING_WEIGHT = 0.0
        module.FORCE_WEIGHT = 0.0
        module.ALM_MULTIPLIERS = np.array([0.4])
        module.ALM_PENALTY = 10.0
        module.MAGNETIC_WELL_WEIGHT = 3.0

        constraint_values = np.array([0.2])
        constraint_grads = [np.array([2.0, -1.0])]
        # ALM and non-ALM branches now both consume the full geometry-penalty
        # bundle (length, cc, cs, curvature, surf_dist, poloidal_extent) so
        # ``physics_total`` is comparable across formulations. Weights for cc,
        # cs, curvature and surf_dist are zero, so those rows do not affect
        # the expected base total.
        objective_eval = {
            "total": 999.0,
            "grad": np.array([99.0, 99.0]),
            "J_QS": 1.5e-4,
            "dJ_QS": np.array([0.1, 0.0]),
            "J_QS_objective": 1.5,
            "dJ_QS_objective": np.array([0.1, 0.0]),
            "J_Boozer": 2.5e-6,
            "dJ_Boozer": np.array([0.0, 0.2]),
            "J_Boozer_objective": 2.5,
            "dJ_Boozer_objective": np.array([0.0, 0.2]),
            "J_iota": 0.7,
            "dJ_iota": np.array([0.3, 0.0]),
            "J_volume": 0.4,
            "dJ_volume": np.array([0.0, 0.5]),
            "J_magnetic_well": 0.8,
            "dJ_magnetic_well": np.array([0.2, 0.3]),
            "J_len": 2.0,
            "dJ_len": np.array([0.6, 0.7]),
            "J_cc": 1.1,
            "dJ_cc": np.array([0.0, 0.0]),
            "J_cs": 0.9,
            "dJ_cs": np.array([0.0, 0.0]),
            "J_curvature": 0.5,
            "dJ_curvature": np.array([0.0, 0.0]),
            "constraint_values": constraint_values,
            "constraint_grads": constraint_grads,
        }

        scalarized = module.apply_frontier_scalarization_override(objective_eval)

        expected_base_total = (
            module.LENGTH_WEIGHT * objective_eval["J_len"]
            + objective_eval["J_QS_objective"]
            + module.EFFECTIVE_RES_WEIGHT * objective_eval["J_Boozer_objective"]
            + module.EFFECTIVE_IOTAS_WEIGHT * objective_eval["J_iota"]
            + module.EFFECTIVE_VOLUME_WEIGHT * objective_eval["J_volume"]
            + module.MAGNETIC_WELL_WEIGHT * objective_eval["J_magnetic_well"]
        )
        expected_base_grad = (
            module.LENGTH_WEIGHT * objective_eval["dJ_len"]
            + objective_eval["dJ_QS_objective"]
            + module.EFFECTIVE_RES_WEIGHT * objective_eval["dJ_Boozer_objective"]
            + module.EFFECTIVE_IOTAS_WEIGHT * objective_eval["dJ_iota"]
            + module.EFFECTIVE_VOLUME_WEIGHT * objective_eval["dJ_volume"]
            + module.MAGNETIC_WELL_WEIGHT * objective_eval["dJ_magnetic_well"]
        )
        expected = module.augmented_inequality_objective(
            expected_base_total,
            expected_base_grad,
            constraint_values,
            constraint_grads,
            module.ALM_MULTIPLIERS,
            module.ALM_PENALTY,
        )

        self.assertAlmostEqual(scalarized["base_total"], expected_base_total)
        self.assertNotEqual(scalarized["base_total"], objective_eval["total"])
        self.assertAlmostEqual(scalarized["total"], expected["total"])
        np.testing.assert_allclose(scalarized["grad"], expected["grad"])

    def test_evaluate_alm_objective_uses_banana_objective_curves_for_spacing_constraints(
        self,
    ):
        module = load_single_stage_example_module()
        zero = FakeAlgebraicObjective(0.0, [0.0])
        all_field_curves = ("tf_curve", "banana_curve", "proxy_curve", "vf_curve")
        banana_objective_curves = ("banana_curve",)
        module.curves = all_field_curves
        module.objective_curves = banana_objective_curves
        module.outer_surface_data = {
            "boozer_surface": SimpleNamespace(surface="outer_surface")
        }
        module.banana_curve = "banana_curve"
        module.banana_curves = banana_objective_curves
        module.banana_curvelengths = [zero]
        module.surface_data = [module.outer_surface_data]
        module.CC_DIST = 0.05
        module.CS_DIST = 0.015
        module.CURVATURE_THRESHOLD = 100.0
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.curvelength = zero
        module.length_target = 1.9
        module.JF = SimpleNamespace(x=np.array([0.0]))
        module.JShear = "shear"
        module.SHEAR_WEIGHT = 9.0
        module.JMagneticWell = "magnetic_well"
        module.MAGNETIC_WELL_WEIGHT = 6.0
        module.args = SimpleNamespace(
            alm_formulation="weighted_sum",
            alm_qs_threshold=1.0,
            alm_boozer_threshold=1.0,
            alm_iota_penalty_threshold=1.0,
            alm_length_penalty_threshold=1.0,
            banana_current_max_A=1.6e4,
        )

        def fake_evaluate_alm_objective_impl(*_args, **kwargs):
            self.assertIs(kwargs["curves"], banana_objective_curves)
            self.assertIsNot(kwargs["curves"], all_field_curves)
            self.assertEqual(kwargs["JShear"], "shear")
            self.assertEqual(kwargs["SHEAR_WEIGHT"], 9.0)
            self.assertEqual(kwargs["JMagneticWell"], "magnetic_well")
            self.assertEqual(kwargs["MAGNETIC_WELL_WEIGHT"], 6.0)
            return {
                "total": 1.0,
                "grad": np.array([0.0]),
                "constraint_values": np.array([]),
                "constraint_grads": [],
            }

        with (
            patch.object(
                module,
                "_evaluate_alm_objective_impl",
                side_effect=fake_evaluate_alm_objective_impl,
            ),
            patch.object(
                module,
                "apply_frontier_scalarization_override",
                side_effect=lambda objective_eval, **_kwargs: objective_eval,
            ),
            patch.object(
                module,
                "single_stage_alm_constraint_names",
                return_value=[],
            ),
            patch.object(
                module,
                "current_single_stage_alm_surface_stack_surfaces",
                return_value=None,
            ),
            patch.object(
                module,
                "current_single_stage_alm_banana_current",
                return_value=None,
            ),
            patch.object(
                module,
                "current_single_stage_alm_banana_currents",
                return_value=None,
            ),
        ):
            result = module.evaluate_alm_objective(
                np.array([1.0]),
                [zero],
                [zero],
                RES_WEIGHT=0.0,
                Jiota=zero,
                IOTAS_WEIGHT=0.0,
                JCurveLength=zero,
                LENGTH_WEIGHT=0.0,
                JCurveCurve=zero,
                JCurveSurface=zero,
                JCurvature=zero,
                multipliers=np.array([]),
                penalty=1.0,
            )

        self.assertAlmostEqual(result["total"], 1.0)

    def test_evaluate_total_objective_matches_raw_impl_outside_frontier_mode(self):
        module = load_single_stage_example_module()
        module.SINGLE_STAGE_GOAL_MODE = "target"
        module.FRONTIER_GOAL_CONFIG = None

        surface_weights = np.array([1.0])
        non_qs = [FakeAlgebraicObjective(1.2, [0.5, 0.0])]
        boozer = [FakeAlgebraicObjective(2.0, [0.0, 0.4])]
        jiota = FakeAlgebraicObjective(-0.1, [-0.3, 0.0])
        curve_length = FakeAlgebraicObjective(0.05, [0.1, 0.1])
        curve_curve = FakeAlgebraicObjective(0.25, [0.3, 0.4])
        curve_surface = FakeAlgebraicObjective(0.15, [0.2, -0.1])
        curvature = FakeAlgebraicObjective(0.35, [0.2, 0.3])
        magnetic_well = FakeAlgebraicObjective(0.45, [0.6, -0.2])
        resolved_terms = {
            "effective_res_weight": 7.0,
            "effective_iotas_weight": 11.0,
            "effective_volume_weight": 0.0,
            "JNonQSObjective": None,
            "JBoozerObjective": None,
            "JVolume": None,
        }

        with patch.object(
            module,
            "resolve_current_surface_objective_terms",
            return_value=resolved_terms,
        ):
            wrapped = module.evaluate_total_objective(
                surface_weights,
                non_qs,
                boozer,
                RES_WEIGHT=999.0,
                Jiota=jiota,
                IOTAS_WEIGHT=888.0,
                JCurveLength=curve_length,
                LENGTH_WEIGHT=0.5,
                JCurveCurve=curve_curve,
                CC_WEIGHT=2.0,
                JCurveSurface=curve_surface,
                CS_WEIGHT=3.0,
                JCurvature=curvature,
                CURVATURE_WEIGHT=4.0,
                JMagneticWell=magnetic_well,
                MAGNETIC_WELL_WEIGHT=5.0,
            )
        raw = module._evaluate_total_objective_impl(
            surface_weights,
            non_qs,
            boozer,
            resolved_terms["effective_res_weight"],
            jiota,
            resolved_terms["effective_iotas_weight"],
            curve_length,
            0.5,
            curve_curve,
            2.0,
            curve_surface,
            3.0,
            curvature,
            4.0,
            JNonQSObjective=resolved_terms["JNonQSObjective"],
            JBoozerObjective=resolved_terms["JBoozerObjective"],
            JVolume=resolved_terms["JVolume"],
            VOLUME_WEIGHT=resolved_terms["effective_volume_weight"],
            JMagneticWell=magnetic_well,
            MAGNETIC_WELL_WEIGHT=5.0,
        )

        self.assertEqual(set(wrapped.keys()), set(raw.keys()))
        for key, raw_value in raw.items():
            wrapped_value = wrapped[key]
            if isinstance(raw_value, np.ndarray):
                np.testing.assert_allclose(wrapped_value, raw_value)
            else:
                self.assertEqual(wrapped_value, raw_value)

    def test_evaluate_total_objective_projects_component_gradients_to_search_space(
        self,
    ):
        module = load_single_stage_example_module()

        surface_weights = np.array([1.0])
        objective_optimizable = object()
        non_qs = [FakeProjectedObjective(1.2, [0.5, 0.0], [0.5, 0.0, 0.0, 0.0])]
        boozer = [FakeProjectedObjective(2.0, [0.0, 0.4], [0.0, 0.4, 0.0, 0.0])]
        jiota = FakeProjectedObjective(-0.1, [-0.3, 0.0], [0.0, 0.0, -0.3, 0.0])
        volume = FakeProjectedObjective(-0.2, [0.0, -0.2], [0.0, 0.0, 0.0, -0.2])
        curve_length = FakeProjectedObjective(0.05, [0.1, 0.1], [0.1, 0.1, 0.0, 0.0])
        curve_curve = FakeProjectedObjective(0.0, [0.0, 0.0], np.zeros(4))
        curve_surface = FakeProjectedObjective(0.0, [0.0, 0.0], np.zeros(4))
        curvature = FakeProjectedObjective(0.0, [0.0, 0.0], np.zeros(4))

        objective_eval = module._evaluate_total_objective_impl(
            surface_weights,
            non_qs,
            boozer,
            RES_WEIGHT=1.0,
            Jiota=jiota,
            IOTAS_WEIGHT=1.0,
            JCurveLength=curve_length,
            LENGTH_WEIGHT=1.0,
            JCurveCurve=curve_curve,
            CC_WEIGHT=0.0,
            JCurveSurface=curve_surface,
            CS_WEIGHT=0.0,
            JCurvature=curvature,
            CURVATURE_WEIGHT=0.0,
            JVolume=volume,
            VOLUME_WEIGHT=1.0,
            objective_optimizable=objective_optimizable,
        )

        self.assertEqual(objective_eval["grad"].shape, (4,))
        self.assertEqual(objective_eval["dJ_QS_objective"].shape, (4,))
        self.assertEqual(objective_eval["dJ_Boozer_objective"].shape, (4,))
        self.assertEqual(objective_eval["dJ_iota"].shape, (4,))
        self.assertEqual(objective_eval["dJ_volume"].shape, (4,))
        np.testing.assert_allclose(objective_eval["dJ_QS"], [0.5, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(objective_eval["dJ_Boozer"], [0.0, 0.4, 0.0, 0.0])
        np.testing.assert_allclose(objective_eval["dJ_iota"], [0.0, 0.0, -0.3, 0.0])
        np.testing.assert_allclose(objective_eval["dJ_volume"], [0.0, 0.0, 0.0, -0.2])

    def test_validate_single_stage_alm_formulation_args_rejects_frontier_thresholded_physics(
        self,
    ):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            alm_formulation="thresholded_physics",
            single_stage_goal_mode="frontier",
            constraint_method="alm",
            alm_qs_threshold=0.1,
            alm_boozer_threshold=0.2,
            alm_iota_penalty_threshold=0.3,
            alm_length_penalty_threshold=0.4,
        )

        with self.assertRaisesRegex(
            ValueError,
            "frontier is not compatible with --alm-formulation=thresholded_physics",
        ):
            module.validate_single_stage_alm_formulation_args(args)

    def test_validate_single_stage_alm_formulation_args_rejects_zero_threshold(self):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            alm_formulation="thresholded_physics",
            single_stage_goal_mode="single_stage",
            constraint_method="alm",
            alm_qs_threshold=0.0,
            alm_boozer_threshold=0.2,
            alm_iota_penalty_threshold=0.3,
            alm_length_penalty_threshold=0.4,
        )

        with self.assertRaisesRegex(
            ValueError,
            r"ALM threshold '--alm-qs-threshold' must be a finite positive value",
        ):
            module.validate_single_stage_alm_formulation_args(args)

    def test_validate_single_stage_alm_formulation_args_rejects_negative_threshold(
        self,
    ):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            alm_formulation="thresholded_physics",
            single_stage_goal_mode="single_stage",
            constraint_method="alm",
            alm_qs_threshold=0.1,
            alm_boozer_threshold=-0.2,
            alm_iota_penalty_threshold=0.3,
            alm_length_penalty_threshold=0.4,
        )

        with self.assertRaisesRegex(
            ValueError,
            r"ALM threshold '--alm-boozer-threshold' must be a finite positive value",
        ):
            module.validate_single_stage_alm_formulation_args(args)

    def test_validate_single_stage_alm_formulation_args_rejects_nan_threshold(self):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            alm_formulation="thresholded_physics",
            single_stage_goal_mode="single_stage",
            constraint_method="alm",
            alm_qs_threshold=0.1,
            alm_boozer_threshold=0.2,
            alm_iota_penalty_threshold=float("nan"),
            alm_length_penalty_threshold=0.4,
        )

        with self.assertRaisesRegex(
            ValueError,
            r"ALM threshold '--alm-iota-penalty-threshold' must be a finite positive value",
        ):
            module.validate_single_stage_alm_formulation_args(args)

    def test_validate_single_stage_alm_formulation_args_accepts_positive_thresholds(
        self,
    ):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            alm_formulation="thresholded_physics",
            single_stage_goal_mode="single_stage",
            constraint_method="alm",
            alm_qs_threshold=1.0e-9,
            alm_boozer_threshold=0.2,
            alm_iota_penalty_threshold=0.3,
            alm_length_penalty_threshold=0.4,
        )
        # Strictly-positive, even very small (real geometric tolerance regime), must pass.
        module.validate_single_stage_alm_formulation_args(args)

    def test_validate_single_stage_alm_formulation_args_rejects_weighted_sum_alm_with_nonzero_length_weight(
        self,
    ):
        # Per .alm_audit/FIX_PLAN.md S1: weighted_sum + ALM owns the coil-length
        # constraint as an inequality, while build_total_objective also pulls the
        # same QuadraticPenalty in via LENGTH_WEIGHT. Combining the two double-feeds
        # the same boundary and corrupts the saved ALM multiplier, so we reject the
        # combination loudly instead of silently zeroing the operator's weight.
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            alm_formulation="weighted_sum",
            single_stage_goal_mode="single_stage",
            constraint_method="alm",
            length_weight=1.0,
            alm_boozer_threshold=0.2,
        )

        with self.assertRaisesRegex(
            ValueError,
            r"ALM weighted_sum formulation owns the coil-length constraint",
        ):
            module.validate_single_stage_alm_formulation_args(args)

    def test_validate_single_stage_alm_formulation_args_accepts_weighted_sum_alm_with_zero_length_weight(
        self,
    ):
        # Setting --length-weight 0 is the documented escape hatch in S1 — it
        # disables the soft penalty path so the ALM constraint is the sole owner.
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            alm_formulation="weighted_sum",
            single_stage_goal_mode="single_stage",
            constraint_method="alm",
            length_weight=0.0,
            alm_boozer_threshold=0.2,
        )
        module.validate_single_stage_alm_formulation_args(args)

    def test_validate_single_stage_alm_formulation_args_thresholded_physics_alm_unaffected_by_length_weight(
        self,
    ):
        # thresholded_physics ALM ignores LENGTH_WEIGHT entirely (see
        # single_stage_objectives.evaluate_base_objective): it returns total=0,
        # grad=0 in that mode, so the double-feed bug cannot trigger and the new
        # check must not engage.
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            alm_formulation="thresholded_physics",
            single_stage_goal_mode="single_stage",
            constraint_method="alm",
            length_weight=1.0,
            alm_qs_threshold=0.1,
            alm_boozer_threshold=0.2,
            alm_iota_penalty_threshold=0.3,
            alm_length_penalty_threshold=0.4,
        )
        module.validate_single_stage_alm_formulation_args(args)

    def test_validate_single_stage_alm_formulation_args_weighted_sum_no_alm_unaffected(
        self,
    ):
        # Pure weighted_sum (no ALM) is the legacy soft-penalty path; LENGTH_WEIGHT
        # is the only owner of the coil-length term, so the new check must not
        # engage. This guards against accidentally widening the raise to penalty
        # mode.
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            alm_formulation="weighted_sum",
            single_stage_goal_mode="single_stage",
            constraint_method="penalty",
            length_weight=1.0,
        )
        module.validate_single_stage_alm_formulation_args(args)

    def test_alm_iota_penalty_threshold_help_documents_squared_penalty_units(self):
        # The CLI help text for --alm-iota-penalty-threshold must pin the actual
        # constraint form 0.5*(iota - iota_target)**2 <= T (with NO iotas_weight
        # factor: see single_stage_objectives._objective_upper_bound_constraint
        # call on Jiota = QuadraticPenalty(..., f='identity')) and the correct
        # operator-facing conversion T = 0.5 * d**2. It must also cross-link to
        # --stage2-iota-tolerance because that flag uses the deviation directly,
        # not the squared penalty units, and operators must not assume the two
        # flags are interchangeable in numerical scale.
        module = load_single_stage_example_module()

        with (
            patch.object(
                sys,
                "argv",
                [
                    "single_stage_banana_example.py",
                    "--help",
                ],
            ),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            with self.assertRaises(SystemExit) as excinfo:
                module.parse_args()

        help_text = stdout.getvalue()
        # argparse hard-wraps long help strings around column ~58, breaking on
        # internal spaces. Collapsing whitespace ("\n   " -> " ") yields the
        # rendered prose minus the wrap geometry. argparse may also break a
        # long flag like --stage2-iota-tolerance at an internal hyphen, leaving
        # "--stage2-iota- tolerance" after collapse; we tolerate that by
        # asserting on the wrap-resilient prefix --stage2-iota- and the
        # adjacent token "tolerance" rather than the brittle full flag.
        help_text_flat = " ".join(help_text.split())
        self.assertEqual(excinfo.exception.code, 0)
        self.assertIn("--alm-iota-penalty-threshold", help_text_flat)
        self.assertIn("squared-penalty units", help_text_flat)
        self.assertIn("--stage2-iota-", help_text_flat)
        self.assertIn("tolerance, which is a direct iota deviation", help_text_flat)
        # Pin the actual constraint formula and conversion. These exact strings
        # appear in the (whitespace-collapsed) help text, so any future drift
        # away from the correct half-squared form fails this assertion.
        self.assertIn("0.5*(iota - iota_target)**2 <= T", help_text_flat)
        self.assertIn("T = 0.5 * d**2", help_text_flat)
        # Pin the worked example numerically so operators always have a sanity
        # check: deviation 0.01 -> threshold 0.5 * 0.01**2 = 5e-5.
        self.assertEqual(0.5 * 0.01**2, 5e-5)
        self.assertIn(
            "d = 0.01 -> T = 0.5 * 0.01**2 = 5e-5",
            help_text_flat,
        )

    def test_alm_iota_penalty_constraint_uses_half_squared_form(self):
        # Pins the math identity at the CONSTRAINT level: the iota_penalty entry
        # in the thresholded_physics ALM constraint set is built by feeding
        # Jiota = QuadraticPenalty(surface_iota_term, iota_target) (with default
        # f='identity') into _objective_upper_bound_constraint, with NO weight
        # factor. The QuadraticPenalty contract from
        # simsopt.objectives.utilities is J() = 0.5 * (obj.J() - cons)**2, so
        # the constraint reduces to 0.5 * (iota - iota_target)**2 <= T. This
        # test guards against future help-text drift by anchoring the formula
        # at the place the operator-facing docs MUST reflect.
        module = load_single_stage_example_module()

        class FixedIotaTerm(Optimizable):
            def __init__(self, value):
                super().__init__(x0=np.array([0.0]))
                self._value = float(value)

            def J(self):
                return self._value

            def dJ(self, partials=False):
                if partials:
                    return lambda _objective: np.array([0.0])
                return np.array([0.0])

        iota_target = 0.42
        deviation = 0.01
        iota_term = FixedIotaTerm(iota_target + deviation)

        Jiota = module.build_single_stage_iota_objective(
            iota_term,
            iota_target,
            goal_mode="target",
        )

        # build_single_stage_iota_objective in target mode is documented as
        # QuadraticPenalty(iota_term, iota_target) with default f='identity'.
        from simsopt.objectives.utilities import QuadraticPenalty

        self.assertIsInstance(Jiota, QuadraticPenalty)
        self.assertEqual(Jiota.f, "identity")
        # The penalty value is exactly 0.5 * deviation**2 -- this is the term
        # bounded by --alm-iota-penalty-threshold in thresholded_physics mode.
        self.assertAlmostEqual(Jiota.J(), 0.5 * deviation**2)
        self.assertAlmostEqual(Jiota.J(), 5e-5)

    def test_single_stage_parse_args_accepts_boozer_stage_refinement_flags(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--boozer-stage-refinement",
                "--refinement-boozer-stage",
                "final",
                "--refinement-maxiter",
                "25",
                "--refinement-chunk-maxiter",
                "7",
                "--refinement-max-stalled-chunks",
                "3",
            ],
        ):
            args = module.parse_args()

        self.assertTrue(args.boozer_stage_refinement)
        self.assertEqual(args.refinement_boozer_stage, "final")
        self.assertEqual(args.refinement_maxiter, 25)
        self.assertEqual(args.refinement_chunk_maxiter, 7)
        self.assertEqual(args.refinement_max_stalled_chunks, 3)

    def test_stage2_parse_args_accepts_tf_current_A(self):
        module = load_stage2_module()

        with patch.object(
            sys, "argv", ["banana_coil_solver.py", "--tf-current-A", "-80000"]
        ):
            args = module.parse_args()

        self.assertEqual(args.tf_current_A, -80000.0)

    def test_stage2_parse_args_accepts_jhalpern_mode_and_flip_banana(self):
        module = load_stage2_module()

        with patch.object(
            sys,
            "argv",
            [
                "banana_coil_solver.py",
                "--finite-current-mode",
                "jhalpern30_proxy_field",
                "--flip-banana",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.finite_current_mode, "jhalpern30_proxy_field")
        self.assertTrue(args.flip_banana)

    def test_stage2_parse_args_accepts_stage2_poloidal_weight(self):
        module = load_stage2_module()

        with patch.object(
            sys,
            "argv",
            ["banana_coil_solver.py", "--stage2-poloidal-weight", "4.5"],
        ):
            args = module.parse_args()

        self.assertEqual(args.stage2_poloidal_weight, 4.5)

    def test_stage2_parse_args_uses_measured_lbfgsb_maxcor_default(self):
        module = load_stage2_module()

        with patch.object(sys, "argv", ["banana_coil_solver.py"]):
            args = module.parse_args()

        self.assertEqual(args.maxcor, module.DEFAULT_LBFGSB_MAXCOR)
        self.assertEqual(args.maxcor, 40)

    def test_stage2_parse_args_exposes_keepout_contract_fields(self):
        with (
            patch.object(sys, "argv", ["banana_coil_solver.py"]),
            patch.dict(os.environ, {}, clear=True),
        ):
            args = stage2_solver.parse_args()

        self.assertEqual(args.stage2_available_envelope_reward_weight, 0.0)
        self.assertEqual(args.stage2_hardware_sdf_free_space_reward_weight, 0.0)
        self.assertEqual(args.stage2_hardware_keepout_backend, "point_cloud")
        self.assertIsNone(args.stage2_hardware_keepout_sdf_manifest)
        self.assertIsNone(args.stage2_plasma_surface_path)
        self.assertEqual(
            args.stage2_hardware_keepout_json,
            stage2_solver.DEFAULT_HARDWARE_KEEPOUT_JSON_PATH,
        )
        self.assertEqual(
            args.stage2_hardware_keepout_glb,
            stage2_solver.DEFAULT_HARDWARE_KEEPOUT_GLB_PATH,
        )

    def _stage2_s_hel_argv(self, *extra_args: str) -> list[str]:
        return [
            "banana_coil_solver.py",
            "--enable-s-hel-objective",
            *extra_args,
        ]

    def test_stage2_parse_args_accepts_explicit_s_hel_objective_schedule(self):
        module = load_stage2_module()

        with patch.object(
            sys,
            "argv",
            self._stage2_s_hel_argv(),
        ):
            args = module.parse_args()

        self.assertTrue(args.enable_s_hel_objective)
        self.assertEqual(args.s_hel_objective_weight, 1.0e-3)

    def test_stage2_build_s_hel_objective_constructs_enabled_schedule(self):
        module = load_stage2_module()
        args = SimpleNamespace(
            enable_s_hel_objective=True,
            s_hel_objective_weight=2.0e-3,
        )
        field = object()
        surface = object()
        calls: list[tuple[object, object]] = []

        class _FakeHelicalFieldContentObjective:
            def __init__(self, received_field, received_surface):
                calls.append((received_field, received_surface))

        with patch.object(
            module,
            "HelicalFieldContentObjective",
            _FakeHelicalFieldContentObjective,
        ):
            objective, weight = module.build_s_hel_objective(args, field, surface)

        self.assertIsInstance(objective, _FakeHelicalFieldContentObjective)
        self.assertEqual(weight, 2.0e-3)
        self.assertEqual(calls, [(field, surface)])

    def test_stage2_parse_args_accepts_s_hel_objective_without_iota_probe(self):
        module = load_stage2_module()

        with patch.object(
            sys,
            "argv",
            ["banana_coil_solver.py", "--enable-s-hel-objective"],
        ):
            args = module.parse_args()

        self.assertTrue(args.enable_s_hel_objective)
        self.assertIsNone(args.stage2_iota_target)

    def test_stage2_parse_args_rejects_nonpositive_s_hel_weight(self):
        module = load_stage2_module()

        with patch.object(
            sys,
            "argv",
            self._stage2_s_hel_argv("--s-hel-objective-weight", "0"),
        ):
            with self.assertRaises(SystemExit):
                module.parse_args()

    def test_stage2_parse_args_rejects_nonfinite_s_hel_weight(self):
        module = load_stage2_module()

        for value in ("nan", "inf"):
            with self.subTest(value=value):
                with patch.object(
                    sys,
                    "argv",
                    self._stage2_s_hel_argv("--s-hel-objective-weight", value),
                ):
                    with self.assertRaises(SystemExit):
                        module.parse_args()

    def test_single_stage_parse_args_preserve_wrapper_default_hardware_thresholds(self):
        module = load_single_stage_example_module()

        with patch.object(sys, "argv", ["single_stage_banana_example.py"]):
            args = module.parse_args()

        self.assertEqual(args.cs_dist, module.COIL_PLASMA_MIN_DIST_M)
        self.assertEqual(args.curvature_threshold, 100.0)
        self.assertEqual(args.banana_current_max_A, 16000.0)
        self.assertEqual(args.single_stage_banana_geometry_mode, "shared_symmetry")
        self.assertEqual(args.single_stage_banana_current_mode, "shared")
        self.assertEqual(args.single_stage_banana_current_coordinate_scaling, "none")
        self.assertEqual(args.alm_qs_threshold, 3.0e-3)
        self.assertEqual(args.alm_boozer_threshold, 1.0e-4)
        self.assertEqual(args.alm_iota_penalty_threshold, 1.0e-4)
        self.assertEqual(args.alm_length_penalty_threshold, 1.0e-4)
        self.assertFalse(args.flip_banana)
        self.assertFalse(args.finite_build)
        self.assertIsNone(args.finitebuild_frame_aware_curvature_threshold)

    def test_single_stage_finite_build_tightens_curvature_threshold_by_default(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            ["single_stage_banana_example.py", "--finite-build"],
        ):
            args = module.parse_args()

        module.validate_single_stage_finite_build_cli_args(args)
        finite_build_settings = module.resolve_single_stage_finite_build_settings(args)
        enabled = module.single_stage_frame_aware_curvature_threshold_enabled(args)
        threshold, pack_limit, applied = (
            module.single_stage_frame_aware_curvature_tightening(
                args.curvature_threshold,
                finite_build_settings,
                enabled,
            )
        )

        self.assertTrue(args.finite_build)
        self.assertTrue(enabled)
        self.assertTrue(applied)
        self.assertLess(threshold, args.curvature_threshold)
        self.assertAlmostEqual(threshold, pack_limit)
        # Adopted self-intersection model: cap = 1/(inner-radius margin +
        # outer-channel corner reach), independent of the conductor-pack grid.
        hardware_contracts = load_hardware_contracts_module()
        expected_limit = 1.0 / (
            hardware_contracts.TYPE_KK_INNER_RADIUS_MARGIN_M
            + np.hypot(
                hardware_contracts.TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
                hardware_contracts.TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
            )
        )
        self.assertAlmostEqual(threshold, expected_limit)

        metadata = module.single_stage_finite_build_metadata(
            finite_build_settings,
            threshold_enabled=enabled,
            threshold_applied=applied,
            pack_limit_inv_m=pack_limit,
            final_max_curvature=0.99 * threshold,
        )
        self.assertTrue(metadata["FINITE_BUILD_ENABLED"])
        self.assertTrue(metadata["FINITEBUILD_CURVATURE_OK"])
        self.assertEqual(metadata["FINITEBUILD_FRAME"], "surface_tangent")

    def test_single_stage_finite_build_can_keep_centerline_curvature_threshold(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--finite-build",
                "--no-finitebuild-frame-aware-curvature-threshold",
            ],
        ):
            args = module.parse_args()

        finite_build_settings = module.resolve_single_stage_finite_build_settings(args)
        enabled = module.single_stage_frame_aware_curvature_threshold_enabled(args)
        threshold, pack_limit, applied = (
            module.single_stage_frame_aware_curvature_tightening(
                args.curvature_threshold,
                finite_build_settings,
                enabled,
            )
        )

        self.assertFalse(enabled)
        self.assertFalse(applied)
        self.assertIsNone(pack_limit)
        self.assertEqual(threshold, args.curvature_threshold)

    def test_single_stage_frame_aware_threshold_requires_finite_build(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--filament-only",
                "--finitebuild-frame-aware-curvature-threshold",
            ],
        ):
            args = module.parse_args()

        with self.assertRaisesRegex(ValueError, "requires --finite-build"):
            module.validate_single_stage_finite_build_cli_args(args)

    def test_single_stage_uses_quartic_curvature_penalty(self):
        module = load_single_stage_example_module()
        source = EXAMPLE_MODULE_PATH.read_text()

        self.assertEqual(module.CURVATURE_P_NORM, 4)
        self.assertIn(
            "LpCurveCurvature(curve, CURVATURE_P, CURVATURE_THRESHOLD)",
            source,
        )
        self.assertIn("for curve in banana_curves", source)
        self.assertIn("CURVATURE_P_NORM", source)
        self.assertIn("COIL_LENGTH_MIN_FRACTION * length_target", source)
        self.assertIn("JCurveLengthMin = average_surface_objectives(", source)

    def test_single_stage_curvature_p_continuation_schedule(self):
        module = load_single_stage_example_module()

        self.assertEqual(
            module.parse_curvature_p_continuation_schedule("", 4),
            (4,),
        )
        self.assertEqual(
            module.parse_curvature_p_continuation_schedule("2, 4,6", 4),
            (2, 4, 6),
        )
        self.assertEqual(module.curvature_p_for_stage("initial", (2, 6)), 2)
        self.assertEqual(module.curvature_p_for_stage("final", (2, 6)), 6)

        with self.assertRaisesRegex(ValueError, "positive integers"):
            module.parse_curvature_p_continuation_schedule("2,bad", 4)
        with self.assertRaisesRegex(ValueError, "values must be >= 1"):
            module.parse_curvature_p_continuation_schedule("2,0", 4)

    def test_single_stage_scalar_hardware_helpers_use_worst_banana_curve(self):
        module = load_single_stage_example_module()

        class _FakeCurve:
            def __init__(self, length, kappa):
                self.length = length
                self._kappa = np.asarray(kappa, dtype=float)

            def kappa(self):
                return self._kappa

        class _FakeCurveLength:
            def __init__(self, curve):
                self.curve = curve

            def J(self):
                return self.curve.length

        curves = (_FakeCurve(1.2, [12.0, 18.0]), _FakeCurve(1.5, [45.0, 72.0]))
        with patch.object(module, "CurveLength", _FakeCurveLength):
            self.assertEqual(module.max_single_stage_banana_curve_length(curves), 1.5)
        self.assertEqual(module.max_single_stage_banana_curvature(curves), 72.0)

    def test_single_stage_parse_args_accepts_flip_banana(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            ["single_stage_banana_example.py", "--flip-banana"],
        ):
            args = module.parse_args()

        self.assertTrue(args.flip_banana)

    def test_single_stage_parse_args_accepts_jhalpern_finite_current_mode(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--finite-current-mode",
                "jhalpern30_proxy_field",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.finite_current_mode, "jhalpern30_proxy_field")

    def test_resolve_single_stage_iota_target_negates_flip_banana(self):
        module = load_single_stage_example_module()

        self.assertEqual(
            module.resolve_single_stage_iota_target(
                SimpleNamespace(iota_target=0.15, flip_banana=False)
            ),
            0.15,
        )
        self.assertEqual(
            module.resolve_single_stage_iota_target(
                SimpleNamespace(iota_target=0.15, flip_banana=True)
            ),
            -0.15,
        )

    def test_resolve_effective_single_stage_iota_target_honors_stage2_flip_metadata(
        self,
    ):
        module = load_single_stage_example_module()
        args = SimpleNamespace(iota_target=0.15, flip_banana=False)

        self.assertEqual(
            module.resolve_effective_single_stage_iota_target(
                args,
                {"FLIP_BANANA": True, "IOTA_TARGET_SIGN": -1},
            ),
            -0.15,
        )
        self.assertEqual(
            module.resolve_effective_single_stage_iota_target(
                args,
                {"FLIP_BANANA": False, "IOTA_TARGET_SIGN": 1},
            ),
            0.15,
        )

    def test_resolve_effective_single_stage_iota_target_rejects_partial_flip_metadata(
        self,
    ):
        module = load_single_stage_example_module()
        args = SimpleNamespace(iota_target=0.15, flip_banana=False)

        with self.assertRaisesRegex(ValueError, "missing FLIP_BANANA"):
            module.resolve_effective_single_stage_iota_target(
                args,
                {"IOTA_TARGET_SIGN": -1},
            )

    def test_resolve_effective_single_stage_iota_target_rejects_conflicting_flip_metadata(
        self,
    ):
        module = load_single_stage_example_module()
        args = SimpleNamespace(iota_target=0.15, flip_banana=False)

        with self.assertRaisesRegex(ValueError, "metadata disagree"):
            module.resolve_effective_single_stage_iota_target(
                args,
                {"FLIP_BANANA": True, "IOTA_TARGET_SIGN": 1},
            )

    def test_flip_banana_banner_reports_effective_iota_target(self):
        module = load_single_stage_example_module()

        self.assertEqual(
            module.format_flip_banana_banner(0.15, -0.15),
            "[FLIP_BANANA] --flip-banana active: iota_target will be negated "
            "(requested 0.15 -> effective -0.15; mirror banana convention).",
        )

    def test_single_stage_startup_prints_flip_banana_banner(self):
        source = EXAMPLE_MODULE_PATH.read_text()

        self.assertIn("print(format_flip_banana_banner(", source)
        self.assertIn("flush=True", source)

    def test_single_stage_parse_args_accepts_independent_banana_current_mode(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--single-stage-banana-current-mode",
                "independent",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.single_stage_banana_current_mode, "independent")

    def test_single_stage_parse_args_accepts_materialized_banana_geometry_mode(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--single-stage-banana-geometry-mode",
                "materialized_cws",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.single_stage_banana_geometry_mode, "materialized_cws")

    def test_single_stage_parse_args_accepts_banana_current_coordinate_scaling(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--single-stage-banana-current-mode",
                "independent",
                "--single-stage-banana-current-coordinate-scaling",
                "seed-relative",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.single_stage_banana_current_mode, "independent")
        self.assertEqual(
            args.single_stage_banana_current_coordinate_scaling,
            "seed-relative",
        )

    def test_validate_single_stage_current_args_rejects_above_hardware_limit(self):
        module = load_single_stage_example_module()

        args = SimpleNamespace(
            banana_current_max_A=20000.0,
            offspec_replay_debug_only=False,
        )

        with self.assertRaisesRegex(ValueError, "banana-current-max-A"):
            module.validate_single_stage_current_args(args)

    def test_validate_single_stage_current_args_allows_offspec_debug_replay_limit(self):
        module = load_single_stage_example_module()

        args = SimpleNamespace(
            banana_current_max_A=20000.0,
            offspec_replay_debug_only=True,
            single_stage_banana_current_mode="shared",
        )

        module.validate_single_stage_current_args(args)

    def test_validate_single_stage_current_args_allows_independent_alm(self):
        module = load_single_stage_example_module()

        args = SimpleNamespace(
            banana_current_max_A=16000.0,
            offspec_replay_debug_only=False,
            single_stage_banana_current_mode="independent",
            constraint_method="alm",
        )

        module.validate_single_stage_current_args(args)

    def test_validate_single_stage_current_args_rejects_invalid_mode(self):
        module = load_single_stage_example_module()

        args = SimpleNamespace(
            banana_current_max_A=16000.0,
            single_stage_banana_current_mode="bogus",
            constraint_method="penalty",
        )

        with self.assertRaisesRegex(
            ValueError,
            "--single-stage-banana-current-mode must be one of \\{shared, independent\\}",
        ):
            module.validate_single_stage_current_args(args)

    def test_validate_single_stage_current_args_rejects_invalid_geometry_mode(self):
        module = load_single_stage_example_module()

        args = SimpleNamespace(
            banana_current_max_A=16000.0,
            single_stage_banana_current_mode="shared",
            single_stage_banana_geometry_mode="bogus",
            constraint_method="penalty",
        )

        with self.assertRaisesRegex(
            ValueError,
            "--single-stage-banana-geometry-mode must be one of "
            "\\{shared_symmetry, materialized_cws\\}",
        ):
            module.validate_single_stage_current_args(args)

    def test_current_single_stage_alm_banana_currents_returns_independent_controls(
        self,
    ):
        module = load_single_stage_example_module()
        original_state = getattr(module, "banana_current_state", None)
        had_state = hasattr(module, "banana_current_state")
        current_a = Current(1.2e4)
        current_b = Current(-1.2e4)
        module.banana_current_state = module.SingleStageBananaCurrentState(
            mode="independent",
            currents=(current_a, current_b),
            seed_currents_A=(1.0e4, -1.0e4),
        )
        try:
            self.assertIsNone(module.current_single_stage_alm_banana_current())
            self.assertEqual(
                module.current_single_stage_alm_banana_currents(),
                (current_a, current_b),
            )
            self.assertEqual(
                module.single_stage_alm_constraint_names(
                    alm_formulation="weighted_sum",
                    banana_current_state=module.banana_current_state,
                ),
                [
                    "coil_coil_spacing",
                    "coil_surface_spacing",
                    "max_curvature",
                    "coil_length_upper_bound",
                    "coil_length_min",
                    "poloidal_extent",
                    "width_min",
                    "width_max",
                    "self_intersect",
                    "lcfs_major_radius",
                    "lcfs_minor_radius",
                    "banana_current_0_upper_bound",
                    "banana_current_1_upper_bound",
                ],
            )
        finally:
            if had_state:
                module.banana_current_state = original_state
            else:
                delattr(module, "banana_current_state")

    def test_apply_default_stage2_seed_args_uses_legacy_seed_defaults(self):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            plasma_surf_filename="wout_nfp22ginsburg_000_014417_iota15.nc",
            stage2_seed_major_radius=None,
            stage2_seed_toroidal_flux=None,
            stage2_seed_length_weight=None,
            stage2_seed_cc_weight=None,
            stage2_seed_curvature_weight=None,
            stage2_seed_cc_threshold=None,
            stage2_seed_curvature_threshold=None,
            stage2_seed_banana_surf_radius=None,
            stage2_seed_tf_current_A=None,
            stage2_seed_order=None,
            stage2_seed_banana_init_current_A=None,
            accept_offspec_r0_seed=False,
        )

        module.apply_default_stage2_seed_args(args)

        self.assertEqual(args.stage2_seed_curvature_threshold, 100.0)
        self.assertEqual(
            args.stage2_seed_banana_surf_radius,
            module.BANANA_WINDING_MINOR_RADIUS_M,
        )
        self.assertEqual(args.stage2_seed_tf_current_A, -8.0e4)
        self.assertEqual(args.stage2_seed_cc_threshold, module.COIL_COIL_MIN_DIST_M)
        self.assertEqual(args.stage2_seed_major_radius, 0.976)
        self.assertEqual(args.stage2_seed_toroidal_flux, 0.24)
        self.assertEqual(args.stage2_seed_banana_init_current_A, -1.0e4)

    def test_apply_default_stage2_seed_args_preserves_cli_overrides(self):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            plasma_surf_filename="wout_nfp22ginsburg_000_014417_iota15.nc",
            stage2_seed_major_radius=None,
            stage2_seed_toroidal_flux=None,
            stage2_seed_length_weight=None,
            stage2_seed_cc_weight=None,
            stage2_seed_curvature_weight=None,
            stage2_seed_cc_threshold=0.06,
            stage2_seed_curvature_threshold=80.0,
            stage2_seed_banana_surf_radius=None,
            stage2_seed_tf_current_A=-75000.0,
            stage2_seed_order=None,
            stage2_seed_banana_init_current_A=None,
            accept_offspec_r0_seed=False,
        )

        module.apply_default_stage2_seed_args(args)

        self.assertEqual(args.stage2_seed_cc_threshold, 0.06)
        self.assertEqual(args.stage2_seed_curvature_threshold, 80.0)
        self.assertEqual(args.stage2_seed_tf_current_A, -75000.0)

    def test_apply_default_stage2_seed_args_uses_launch_tf_current(self):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            plasma_surf_filename="wout_nfp22ginsburg_000_014417_iota15.nc",
            stage2_seed_major_radius=None,
            stage2_seed_toroidal_flux=None,
            stage2_seed_length_weight=None,
            stage2_seed_cc_weight=None,
            stage2_seed_curvature_weight=None,
            stage2_seed_cc_threshold=None,
            stage2_seed_curvature_threshold=None,
            stage2_seed_banana_surf_radius=None,
            stage2_seed_tf_current_A=None,
            stage2_seed_order=None,
            stage2_seed_banana_init_current_A=None,
            accept_offspec_r0_seed=False,
            tf_current_A=-70000.0,
        )

        module.apply_default_stage2_seed_args(args)

        self.assertEqual(args.stage2_seed_tf_current_A, -70000.0)

    def test_apply_default_stage2_seed_args_allows_offspec_debug_tf_current(self):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            plasma_surf_filename="wout_nfp22ginsburg_000_014417_iota15.nc",
            stage2_seed_major_radius=0.915,
            stage2_seed_toroidal_flux=None,
            stage2_seed_length_weight=None,
            stage2_seed_cc_weight=None,
            stage2_seed_curvature_weight=None,
            stage2_seed_cc_threshold=None,
            stage2_seed_curvature_threshold=None,
            stage2_seed_banana_surf_radius=None,
            stage2_seed_tf_current_A=None,
            stage2_seed_order=None,
            stage2_seed_banana_init_current_A=None,
            accept_offspec_r0_seed=True,
            offspec_replay_debug_only=True,
            tf_current_A=100000.0,
        )

        module.apply_default_stage2_seed_args(args)

        self.assertEqual(args.stage2_seed_tf_current_A, 100000.0)
        self.assertEqual(args.stage2_seed_major_radius, 0.915)

    def test_apply_default_stage2_seed_args_rejects_conflicting_tf_current(self):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            plasma_surf_filename="wout_nfp22ginsburg_000_014417_iota15.nc",
            stage2_seed_major_radius=None,
            stage2_seed_toroidal_flux=None,
            stage2_seed_length_weight=None,
            stage2_seed_cc_weight=None,
            stage2_seed_curvature_weight=None,
            stage2_seed_cc_threshold=None,
            stage2_seed_curvature_threshold=None,
            stage2_seed_banana_surf_radius=None,
            stage2_seed_tf_current_A=-80000.0,
            stage2_seed_order=None,
            stage2_seed_banana_init_current_A=None,
            accept_offspec_r0_seed=False,
            tf_current_A=-70000.0,
        )

        with self.assertRaisesRegex(
            ValueError,
            "--tf-current-A and --stage2-seed-tf-current-A must match",
        ):
            module.apply_default_stage2_seed_args(args)

    def test_apply_default_stage2_seed_args_rejects_offspec_major_radius(self):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            plasma_surf_filename="wout_nfp22ginsburg_000_014417_iota15.nc",
            stage2_seed_major_radius=0.80,
            stage2_seed_toroidal_flux=None,
            stage2_seed_length_weight=None,
            stage2_seed_cc_weight=None,
            stage2_seed_curvature_weight=None,
            stage2_seed_cc_threshold=None,
            stage2_seed_curvature_threshold=None,
            stage2_seed_banana_surf_radius=None,
            stage2_seed_tf_current_A=None,
            stage2_seed_order=None,
            stage2_seed_banana_init_current_A=None,
            accept_offspec_r0_seed=False,
        )

        with self.assertRaisesRegex(ValueError, "vacuum-vessel major radius"):
            module.apply_default_stage2_seed_args(args)

    def test_apply_default_stage2_seed_args_accepts_explicit_offspec_replay_radius(
        self,
    ):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            plasma_surf_filename="wout_nfp22ginsburg_000_014417_iota15.nc",
            stage2_seed_major_radius=0.80,
            stage2_seed_toroidal_flux=None,
            stage2_seed_length_weight=None,
            stage2_seed_cc_weight=None,
            stage2_seed_curvature_weight=None,
            stage2_seed_cc_threshold=None,
            stage2_seed_curvature_threshold=None,
            stage2_seed_banana_surf_radius=None,
            stage2_seed_tf_current_A=None,
            stage2_seed_order=None,
            stage2_seed_banana_init_current_A=None,
            accept_offspec_r0_seed=True,
        )

        module.apply_default_stage2_seed_args(args)

        self.assertEqual(args.stage2_seed_major_radius, 0.80)

    def test_single_stage_resume_contract_accepts_offspec_radius_only_when_explicit(
        self,
    ):
        module = load_single_stage_example_module()
        results = {
            "MAJOR_RADIUS": 0.80,
            "TOROIDAL_FLUX": 0.24,
            "banana_surf_radius": 0.21,
        }

        with self.assertRaisesRegex(ValueError, "vacuum-vessel major radius"):
            module.validate_single_stage_resume_seed_contract(
                results,
                accept_offspec_r0_seed=False,
            )

        module.validate_single_stage_resume_seed_contract(
            results,
            accept_offspec_r0_seed=True,
        )

    def test_stage2_parse_args_accepts_banana_current_controls(self):
        module = load_stage2_module()

        with patch.object(
            sys,
            "argv",
            [
                "banana_coil_solver.py",
                "--banana-init-current-A",
                "-12000",
                "--banana-current-max-A",
                "16000",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.banana_init_current_A, -12000.0)
        self.assertEqual(args.banana_current_max_A, 16000.0)

    def test_stage2_validate_banana_current_cli_args_rejects_above_hardware_limit(self):
        module = load_stage2_module()

        args = SimpleNamespace(
            banana_init_current_A=18000.0,
            banana_current_max_A=20000.0,
            tf_current_A=-80000.0,
            accept_offspec_banana_current_sign=False,
            accept_offspec_banana_current_max=False,
            accept_offspec_tf_current_sign=False,
            accept_offspec_tf_current_magnitude=False,
        )

        with self.assertRaisesRegex(ValueError, "banana-init-current-A"):
            module.validate_banana_current_cli_args(args)

    def test_penalty_traversal_helper_applies_symmetric_box_bound(self):
        module = load_stage2_module()
        leaf_current, scaled_current = _make_unbounded_scaled_current()

        resolved = module.apply_penalty_traversal_forbidden_box_bounds(
            bound_targets={"banana_current": scaled_current},
            requested_thresholds={"banana_current": 16000.0},
        )

        self.assertEqual(resolved, {"banana_current": 16000.0})
        np.testing.assert_allclose(leaf_current.local_lower_bounds, [-16000.0])
        np.testing.assert_allclose(leaf_current.local_upper_bounds, [16000.0])

    def test_stage2_penalty_traversal_can_preserve_positive_seed_sign(self):
        module = load_stage2_module()
        leaf_current, scaled_current = _make_unbounded_scaled_current()

        resolved = module.apply_penalty_traversal_forbidden_box_bounds(
            bound_targets={"banana_current": scaled_current},
            requested_thresholds={"banana_current": 16000.0},
            seed_values={"banana_current": 10000.0},
            preserve_seed_sign_names=frozenset({"banana_current"}),
        )

        self.assertEqual(resolved, {"banana_current": 16000.0})
        np.testing.assert_allclose(leaf_current.local_lower_bounds, [0.0])
        np.testing.assert_allclose(leaf_current.local_upper_bounds, [16000.0])

    def test_stage2_penalty_traversal_can_preserve_negative_seed_sign(self):
        module = load_stage2_module()
        leaf_current, scaled_current = _make_unbounded_scaled_current()

        resolved = module.apply_penalty_traversal_forbidden_box_bounds(
            bound_targets={"banana_current": scaled_current},
            requested_thresholds={"banana_current": 16000.0},
            seed_values={"banana_current": -10000.0},
            preserve_seed_sign_names=frozenset({"banana_current"}),
        )

        self.assertEqual(resolved, {"banana_current": 16000.0})
        np.testing.assert_allclose(leaf_current.local_lower_bounds, [-16000.0])
        np.testing.assert_allclose(leaf_current.local_upper_bounds, [0.0])

    def test_shared_penalty_traversal_helper_uses_schema_bound(self):
        module = load_stage2_module()
        leaf_current, scaled_current = _make_unbounded_scaled_current()

        resolved = module.apply_penalty_traversal_forbidden_box_bounds(
            bound_targets={"banana_current": scaled_current},
            requested_thresholds={"banana_current": 20000.0},
        )

        self.assertEqual(resolved, {"banana_current": 16000.0})
        np.testing.assert_allclose(leaf_current.local_lower_bounds, [-16000.0])
        np.testing.assert_allclose(leaf_current.local_upper_bounds, [16000.0])

    def test_shared_penalty_traversal_helper_rejects_missing_target(self):
        module = load_stage2_module()

        with self.assertRaisesRegex(
            KeyError,
            "Missing penalty box-bound target for hardware constraint 'banana_current'",
        ):
            module.apply_penalty_traversal_forbidden_box_bounds(
                bound_targets={},
                requested_thresholds={"banana_current": 16000.0},
            )


class Stage2ArtifactWriterTests(unittest.TestCase):
    def _fake_constraint_metadata(self):
        return {
            "CONSTRAINT_PROFILE": "unit-test",
            "EFFECTIVE_VALUES": {"TF_CURRENT_A": -80000.0},
            "OVERRIDE_REASON": None,
            "CONTRACT_HASH": "deadbeef",
            "CONTRACT_SCHEMA_VERSION": 1,
        }

    def test_materialize_stage2_artifact_results_rejects_biot_savart_partition_mismatch(
        self,
    ):
        module = load_stage2_module()

        fake_bs = SimpleNamespace(
            coils=[object(), object(), object()],
            save=lambda *_args, **_kwargs: None,
        )

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            self.assertRaisesRegex(
                ValueError,
                "Stage 2 artifact writer partition metadata does not match the loaded BiotSavart coil count",
            ),
        ):
            module.materialize_stage2_artifact_results(
                args=SimpleNamespace(),
                stage2_bs_artifact_path=str(Path(tmpdir) / "biot_savart_opt.json"),
                results_kwargs={
                    "num_tf_coils": 1,
                    "num_banana_coils": 1,
                    "num_proxy_coils": 0,
                    "num_vf_coils": 0,
                },
                stage2_iota_runtime=None,
                new_bs=fake_bs,
                new_surf=SimpleNamespace(),
                constraint_metadata=self._fake_constraint_metadata(),
            )

    def test_materialize_stage2_artifact_results_requires_constraint_metadata(self):
        module = load_stage2_module()

        fake_bs = SimpleNamespace(
            coils=[object()],
            save=lambda *_args, **_kwargs: None,
        )

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            self.assertRaisesRegex(
                ValueError,
                "requires constraint_metadata",
            ),
        ):
            module.materialize_stage2_artifact_results(
                args=SimpleNamespace(),
                stage2_bs_artifact_path=str(Path(tmpdir) / "biot_savart_opt.json"),
                results_kwargs={
                    "num_tf_coils": 1,
                    "num_banana_coils": 0,
                    "num_proxy_coils": 0,
                    "num_vf_coils": 0,
                },
                stage2_iota_runtime=None,
                new_bs=fake_bs,
                new_surf=SimpleNamespace(),
                constraint_metadata=None,
            )

    def test_materialize_stage2_artifact_results_emits_matching_checksum(self):
        module = load_stage2_module()

        def _save(path):
            Path(path).write_text('{"coils": [1, 2]}', encoding="utf-8")

        fake_bs = SimpleNamespace(
            coils=[object(), object()],
            save=_save,
        )

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(
                module,
                "_magnetic_field_plots",
                return_value=0.125,
            ),
            patch.object(
                module,
                "_build_stage2_results_impl",
                return_value={"FIELD_ERROR": 0.125},
            ),
            patch.object(
                module,
                "build_stage2_iota_report_payload",
                return_value={},
            ),
        ):
            artifact_path = Path(tmpdir) / "biot_savart_opt.json"
            constraint_metadata = self._fake_constraint_metadata()
            results = module.materialize_stage2_artifact_results(
                args=SimpleNamespace(),
                stage2_bs_artifact_path=str(artifact_path),
                results_kwargs={
                    "num_tf_coils": 1,
                    "num_banana_coils": 1,
                    "num_proxy_coils": 0,
                    "num_vf_coils": 0,
                },
                stage2_iota_runtime=None,
                new_bs=fake_bs,
                new_surf=SimpleNamespace(),
                constraint_metadata=constraint_metadata,
            )
            expected_digest = module.compute_stage2_bs_sha256(artifact_path)

        self.assertEqual(
            results["STAGE2_BS_SHA256"],
            expected_digest,
        )
        self.assertEqual(results["CONTRACT_HASH"], "deadbeef")
        self.assertEqual(results["CONSTRAINT_PROFILE"], "unit-test")
        self.assertEqual(results["CONTRACT_SCHEMA_VERSION"], 1)

    def test_materialize_stage2_artifact_results_saves_warm_start_boozer_surface(self):
        module = load_stage2_module()

        def _save_bs(path):
            Path(path).write_text('{"coils": [1, 2]}', encoding="utf-8")

        def _save_boozer_surface(path):
            Path(path).write_text('{"surface": "warm"}', encoding="utf-8")

        fake_bs = SimpleNamespace(
            coils=[object(), object()],
            save=_save_bs,
        )
        runtime = SimpleNamespace(
            boozer_surface=SimpleNamespace(
                save=_save_boozer_surface,
                res={"iota": 0.2, "G": -0.377},
            )
        )

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(
                module,
                "_magnetic_field_plots",
                return_value=0.125,
            ),
            patch.object(
                module,
                "_build_stage2_results_impl",
                return_value={"FIELD_ERROR": 0.125},
            ),
            patch.object(
                module,
                "build_stage2_iota_report_payload",
                return_value={},
            ) as report_payload,
        ):
            artifact_path = Path(tmpdir) / "biot_savart_opt.json"
            results = module.materialize_stage2_artifact_results(
                args=SimpleNamespace(),
                stage2_bs_artifact_path=str(artifact_path),
                results_kwargs={
                    "num_tf_coils": 1,
                    "num_banana_coils": 1,
                    "num_proxy_coils": 0,
                    "num_vf_coils": 0,
                },
                stage2_iota_runtime=runtime,
                new_bs=fake_bs,
                new_surf=SimpleNamespace(),
                constraint_metadata=self._fake_constraint_metadata(),
            )
            warm_start_path = artifact_path.with_name("surf_opt_boozer_surface.json")
            self.assertEqual(results["FIELD_ERROR"], 0.125)
            self.assertEqual(
                warm_start_path.read_text(encoding="utf-8"),
                '{"surface": "warm"}',
            )
            state_path = warm_start_path.with_name("surf_opt_boozer_state.json")
            state_payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state_payload["schema_version"], 1)
            self.assertAlmostEqual(state_payload["iota"], 0.2)
            self.assertAlmostEqual(state_payload["G"], -0.377)
            self.assertEqual(
                report_payload.call_args.kwargs["stage2_seed_surf_path"],
                warm_start_path,
            )


class Stage2RuntimeSmokeTests(unittest.TestCase):
    _EXPECTED_BASIN_TELEMETRY = {
        "basin_accepted_hops": 1,
        "basin_rejected_hops": 1,
        "basin_completed_hops": 2,
        "basin_best_objective": 0.42,
        "basin_initial_objective": 0.55,
        "basin_best_hop_objective": 0.42,
        "basin_best_hop_index": 1,
        "basin_best_result_source": "hop",
        "basin_objective_improvement": 0.13,
        "basin_accept_test_rejections": 1,
        "basin_accept_test_triggered": True,
        "basin_nonfinite_rejections": 0,
        "basin_normalized_step_rejections": 1,
    }

    @staticmethod
    def _make_fake_tf_coils(curve_cls, current_cls, *, count=20, current_A=-8.0e4):
        return [
            SimpleNamespace(curve=curve_cls(), current=current_cls(current_A))
            for _ in range(count)
        ]

    def _make_stage2_args(
        self,
        output_root,
        *,
        stage2_hardware_keepout_glb_default,
        **overrides,
    ):
        defaults = {
            "plasma_surf_filename": "demo.nc",
            "equilibria_dir": str(output_root),
            "equilibrium_path": str(Path(output_root) / "demo.nc"),
            "output_root": str(output_root),
            "stage2_bs_path": str(Path(output_root) / "seed.json"),
            "stage2_seed_current_traversal": False,
            "flip_banana": False,
            "nphi": 8,
            "ntheta": 8,
            "init_only": True,
            "banana_surf_radius": 0.21,
            "winding_surface_free_mpol": 0,
            "winding_surface_free_ntor": 0,
            "tf_current_A": -8.0e4,
            "banana_init_current_A": -1.0e4,
            "banana_current_max_A": 1.6e4,
            "vf_current_max_A": 1.6e4,
            "major_radius": 0.976,
            "toroidal_flux": 0.24,
            "order": 2,
            "maxiter": 30,
            "maxcor": 40,
            "ftol": 1e-15,
            "gtol": 1e-15,
            "constraint_method": "penalty",
            "alm_max_outer_iters": 7,
            "alm_penalty_init": 2.0,
            "alm_penalty_scale": 3.0,
            "alm_penalty_max": 50.0,
            "alm_feas_tol": 1e-4,
            "alm_stationarity_tol": 2e-4,
            "alm_trust_radius_init": 0.15,
            "alm_trust_radius_min": 1e-3,
            "alm_trust_radius_shrink": 0.4,
            "alm_trust_radius_grow": 1.8,
            "alm_max_inner_attempts": 5,
            "alm_max_subproblem_continuations": 9,
            "alm_distance_smoothing": 0.005,
            "alm_curvature_smoothing": 0.05,
            "alm_fix_signal_mismatch_guard": False,
            "alm_taylor_test": False,
            "alm_taylor_test_seed": 123,
            "stage2_iota_target": None,
            "stage2_iota_objective_mode": "report",
            "stage2_iota_tolerance": 5.0e-3,
            "stage2_iota_vol_target": 0.10,
            "stage2_iota_constraint_weight": 1.0,
            "stage2_iota_num_tf_coils": 20,
            "stage2_iota_nphi": 91,
            "stage2_iota_ntheta": 32,
            "stage2_iota_mpol": 8,
            "stage2_iota_ntor": 6,
            "length_weight": 5e-4,
            "length_min_weight": 1.0,
            "length_target": 1.9,
            "target_lcfs_max_major_radius_m": in_bounds_lcfs_major_radius_m(),
            "target_lcfs_max_minor_radius_m": in_bounds_lcfs_minor_radius_m(),
            "stage2_plasma_scaling_mode": "lcfs",
            "stage2_plasma_surface_path": None,
            "accept_offspec_major_radius": False,
            "accept_offspec_winding_radius": False,
            "accept_offspec_banana_current_sign": False,
            "accept_offspec_banana_current_max": False,
            "accept_offspec_tf_current_sign": False,
            "accept_offspec_tf_current_magnitude": False,
            "cc_threshold": 0.05,
            "cc_weight": 100.0,
            "curvature_weight": 1e-4,
            "curvature_threshold": 100.0,
            "curvature_p_norm": 4,
            "squared_flux_weight": 1.0,
            "stage2_poloidal_weight": 1.0,
            "stage2_width_weight": 1.0,
            "stage2_selfint_weight": 1.0,
            "self_envelope_mode": "hinge",
            "self_envelope_weight": 1.0,
            "self_distance_window": 0.060,
            "self_envelope_sampling_margin": 0.0,
            "self_envelope_groc_radius_floor": 0.0231,
            "fold_weight": 1.0,
            "fold_geodesic_curvature_limit": 43.3114,
            "fold_geodesic_curvature_margin_fraction": 0.10,
            "stage2_vessel_keepout_weight": 0.0,
            "stage2_available_envelope_reward_weight": 0.0,
            "stage2_hardware_sdf_free_space_reward_weight": 0.0,
            "stage2_hardware_keepout_weight": 0.0,
            "stage2_hardware_keepout_backend": "point_cloud",
            "stage2_hardware_keepout_json": None,
            "stage2_hardware_keepout_glb": stage2_hardware_keepout_glb_default,
            "stage2_hardware_keepout_sdf_manifest": None,
            "stage2_resonant_flux_weight": 0.0,
            "stage2_resonant_iota_target": None,
            "stage2_resonant_delta": 0.02,
            "stage2_resonant_qmax": 8,
            "basin_hops": 0,
            "basin_stepsize": 0.01,
            "basin_temperature": 2.5,
            "basin_niter_success": 6,
            "basin_seed": 7,
            "theta_center": np.pi,
            "phi_center": np.pi / 4.0,
            "theta_width": np.pi / 6.0,
            "phi_width": np.pi / 8.0,
            "num_quadpoints": 16,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _run_stage2_main(
        self,
        *,
        init_only,
        constraint_method,
        use_seed,
        # Seeded smoke tests emulate a valid donor sidecar by default. Opt out
        # only when exercising the explicit sidecar-required rejection path.
        seed_has_results_sidecar=True,
        basin_hops=0,
        banana_current_A=9500.0,
        self_envelope_min_dist=0.08,
        fold_geodesic_curvature_max=30.0,
        alm_accepted_candidate_x=None,
        artifact_state_by_x=None,
        seed_stage2_results=None,
        arg_overrides=(),
        missing_attr_names=(),
    ):
        module = load_stage2_module()
        runtime = {
            "seed_loads": 0,
            "initialize_calls": 0,
            "minimize_calls": 0,
            "minimize_alm_calls": 0,
            "run_basin_hopping_calls": 0,
            "default_maxcor": module.DEFAULT_LBFGSB_MAXCOR,
            "minimize_bounds": None,
            "minimize_options": None,
            "minimize_alm_options": None,
            "basin_bounds": None,
            "basin_options": None,
            "initialize_extra_kwargs": None,
            "curve_curve_curves": None,
            "curve_curve_minimum_distances": [],
            "curve_surface_curves": None,
            "curve_surface_surface_label": None,
            "projected_ellipse_width_args": None,
            "surface_surface_min_distance_labels": None,
            "plasma_geometry_args": None,
            "build_hbt_reference_surface_kwargs": None,
            "stage2_iota_probes": 0,
            "stage2_iota_probe_kwargs": None,
            "stage2_iota_runtime_calls": 0,
            "stage2_iota_runtime_kwargs": None,
            "vessel_keepout_args": None,
            "hardware_keepout_loader_args": None,
            "hardware_keepout_metadata_args": None,
            "hardware_keepout_objective_args": None,
            "results": None,
        }

        class FakeStage2Objective:
            def __init__(self, value, gradient, x=None):
                self._value = float(value)
                self._gradient = np.asarray(gradient, dtype=float)
                self.x = (
                    np.zeros(2, dtype=float)
                    if x is None
                    else np.asarray(x, dtype=float)
                )
                self.lower_bounds = np.full(self.x.shape, -np.inf, dtype=float)
                self.upper_bounds = np.full(self.x.shape, np.inf, dtype=float)
                self.field = SimpleNamespace(clear_cached_properties=lambda: None)

            def J(self):
                return self._value

            def recompute_bell(self):
                return None

            def dJ(self, partials=False):
                if partials:
                    return lambda _objective: self._gradient.copy()
                return self._gradient.copy()

            def __add__(self, other):
                if other == 0:
                    return self
                return FakeStage2Objective(
                    self._value + other.J(),
                    self._gradient + other.dJ(),
                    self.x.copy(),
                )

            __radd__ = __add__

            def __mul__(self, scalar):
                return FakeStage2Objective(
                    self._value * float(scalar),
                    self._gradient * float(scalar),
                    self.x.copy(),
                )

            __rmul__ = __mul__

        class FakeCurrent:
            def __init__(self, value):
                self._value = float(value)
                self.local_lower_bounds = np.array([-np.inf], dtype=float)
                self.local_upper_bounds = np.array([np.inf], dtype=float)

            def __mul__(self, scalar):
                return FakeCurrent(self._value * float(scalar))

            __rmul__ = __mul__

            def fix_all(self):
                return None

            def get_value(self):
                return self._value

        class FakeCurve:
            def fix_all(self):
                return None

        class FakeSurface:
            def __init__(self, *, label, gamma_value, major_radius, minor_radius):
                self.label = label
                self.nfp = 22
                self._gamma_value = float(gamma_value)
                self._major_radius = float(major_radius)
                self._minor_radius = float(minor_radius)

            def gamma(self):
                return np.ones((2, 2, 3), dtype=float) * self._gamma_value

            def unitnormal(self):
                return np.ones((2, 2, 3), dtype=float)

            def to_vtk(self, *_args, **_kwargs):
                return None

            def volume(self):
                return 0.12

            def major_radius(self):
                return self._major_radius

            def minor_radius(self):
                return self._minor_radius

        class FakeBiotSavart:
            def __init__(self):
                self.points = np.zeros((4, 3), dtype=float)
                self.coils = []

            def set_points(self, points):
                self.points = np.asarray(points, dtype=float)

            def B(self):
                return np.ones_like(self.points)

            def save(self, *_args, **_kwargs):
                return None

        class FakeCurveDistance(FakeStage2Objective):
            def __init__(self):
                super().__init__(0.25, [0.3, 0.4])
                self.minimum_distance = 0.05
                self.curves = ["curve_a", "curve_b"]

            def shortest_distance(self):
                return 0.06

        class FakeCurvatureObjective(FakeStage2Objective):
            def __init__(self):
                super().__init__(0.35, [0.2, 0.3])
                self.threshold = 40.0
                self.curve = SimpleNamespace(
                    kappa=lambda: np.array([39.0, 41.0], dtype=float)
                )

        class FakeCurveSurfaceDistance(FakeStage2Objective):
            def __init__(self, curves, surface, minimum_distance):
                super().__init__(0.15, [0.05, 0.06])
                self.minimum_distance = minimum_distance
                self.curves = curves
                self.surface = surface

            def shortest_distance(self):
                return 0.02

        class FakeProjectedEllipseWidth(FakeStage2Objective):
            def __init__(self):
                super().__init__(0.10, [0.0, 0.0])

        class FakeCurveSelfIntersect(FakeStage2Objective):
            def __init__(self):
                super().__init__(0.0, [0.0, 0.0])

            def shortest_self_distance(self):
                return 0.5

        class FakeCurveSelfDistance(FakeStage2Objective):
            def __init__(self):
                super().__init__(0.0, [0.0, 0.0])

            def shortest_self_distance(self):
                return float(self_envelope_min_dist)

            def shortest_groc(self):
                return 0.04

        class FakeCurveSurfaceGeodesicCurvature(FakeStage2Objective):
            def __init__(self):
                super().__init__(0.0, [0.0, 0.0])

            def max_abs_geodesic_curvature(self):
                return float(fold_geodesic_curvature_max)

            def max_abs_frame_binormal_curvature(self):
                # Mirror CurveSurfaceGeodesicCurvature, where this is an alias of
                # max_abs_geodesic_curvature (fold_buildability.py).
                return self.max_abs_geodesic_curvature()

        class FakeVesselEnvelopeKeepout(FakeStage2Objective):
            def __init__(self, curves, *, winding_r0):
                super().__init__(0.03, [0.0, 0.0])
                runtime["vessel_keepout_args"] = {
                    "curves": tuple(curves),
                    "winding_r0": winding_r0,
                }

            def shortest_clearance(self):
                return 0.041

        class FakeAvailableEnvelopeReward(FakeStage2Objective):
            def __init__(self, curves, *, winding_r0):
                super().__init__(-0.04, [0.0, 0.0])
                runtime["available_envelope_reward_args"] = {
                    "curves": tuple(curves),
                    "winding_r0": winding_r0,
                }

        class FakeHardwareKeepout(FakeStage2Objective):
            def __init__(
                self,
                curves,
                points,
                minimum_distance,
                point_weight,
                *,
                winding_r0,
            ):
                super().__init__(0.07, [0.0, 0.0])
                runtime["hardware_keepout_objective_args"] = {
                    "curves": tuple(curves),
                    "points": np.asarray(points, dtype=float),
                    "minimum_distance": minimum_distance,
                    "point_weight": point_weight,
                    "winding_r0": winding_r0,
                }

            def shortest_distance(self):
                return 0.042

        fake_bs = FakeBiotSavart()
        fake_working_surface = FakeSurface(
            label="working",
            gamma_value=0.0,
            major_radius=0.88,
            minor_radius=0.12,
        )
        fake_lcfs_surface = FakeSurface(
            label="lcfs",
            gamma_value=0.2,
            major_radius=in_bounds_lcfs_major_radius_m(),
            minor_radius=in_bounds_lcfs_minor_radius_m(),
        )
        fake_plasma_geometry = SimpleNamespace(
            working_surface=fake_working_surface,
            lcfs_surface=fake_lcfs_surface,
            working_major_radius_m=fake_working_surface.major_radius(),
            working_minor_radius_m=fake_working_surface.minor_radius(),
            lcfs_major_radius_m=fake_lcfs_surface.major_radius(),
            lcfs_minor_radius_m=fake_lcfs_surface.minor_radius(),
            scale_factor=0.75,
        )
        fake_vv = SimpleNamespace(
            label="vv",
            gamma=lambda: np.ones((2, 2, 3), dtype=float) * 0.1,
            to_vtk=lambda *_a, **_k: None,
        )
        fake_banana_curve = SimpleNamespace(
            order=2,
            kappa=lambda: np.array([39.0, 41.0], dtype=float),
            gamma=lambda: np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.1, 0.1, 0.0],
                    [0.0, 0.1, 0.0],
                ],
                dtype=float,
            ),
        )
        fake_banana_coils = [
            SimpleNamespace(
                curve=fake_banana_curve, current=FakeCurrent(banana_current_A)
            )
        ]
        fake_tf_coils = self._make_fake_tf_coils(FakeCurve, FakeCurrent)

        def seed_results_payload():
            return (
                {
                    "FINITE_CURRENT_MODE": "wataru_proxy_field",
                    "NUM_PROXY_COILS": 1,
                    "NUM_VF_COILS": 1,
                }
                if seed_stage2_results is None
                else dict(seed_stage2_results)
            )

        def build_coil_bundle(*, num_proxy_coils, num_vf_coils):
            proxy_coils = [
                SimpleNamespace(
                    curve=f"proxy_curve_{index}",
                    current=FakeCurrent(9000.0),
                )
                for index in range(int(num_proxy_coils))
            ]
            vf_coils = [
                SimpleNamespace(
                    curve=f"vf_curve_{index}",
                    current=FakeCurrent(-500.0),
                )
                for index in range(int(num_vf_coils))
            ]
            curves = [
                *(coil.curve for coil in fake_tf_coils),
                fake_banana_curve,
                *(coil.curve for coil in proxy_coils),
                *(coil.curve for coil in vf_coils),
            ]
            return curves, proxy_coils, vf_coils

        def fake_seed_loader(seed_bs_path, surf, num_tf_coils, out_dir, **_kwargs):
            runtime["seed_loads"] += 1
            self.assertEqual(num_tf_coils, 20)
            self.assertIs(surf, fake_working_surface)
            seed_results = seed_results_payload()
            curves, proxy_coils, vf_coils = build_coil_bundle(
                num_proxy_coils=seed_results.get("NUM_PROXY_COILS", 0),
                num_vf_coils=seed_results.get("NUM_VF_COILS", 0),
            )
            fake_bs.coils = [
                *fake_tf_coils,
                *fake_banana_coils,
                *proxy_coils,
                *vf_coils,
            ]
            return (
                fake_bs,
                curves,
                fake_banana_curve,
                fake_banana_coils,
                fake_tf_coils,
                proxy_coils,
                vf_coils,
            )

        def fake_initialize_coils(
            surf,
            surf_coils,
            tf_coils,
            num_quadpoints,
            order,
            banana_init_current_A,
            phi_center,
            theta_center,
            phi_width,
            theta_width,
            out_dir,
            **extra_kwargs,
        ):
            runtime["initialize_calls"] += 1
            runtime["initialize_extra_kwargs"] = dict(extra_kwargs)
            self.assertIs(surf, fake_working_surface)
            self.assertEqual(surf_coils, "surf_coils")
            self.assertEqual(len(tf_coils), 20)
            self.assertEqual(num_quadpoints, 16)
            self.assertEqual(order, 2)
            self.assertEqual(banana_init_current_A, -1.0e4)
            self.assertEqual(phi_center, np.pi / 4.0)
            self.assertEqual(theta_center, np.pi)
            self.assertEqual(phi_width, np.pi / 8.0)
            self.assertEqual(theta_width, np.pi / 6.0)
            self.assertTrue(str(out_dir).endswith("outputs-demo.nc/"))
            # Fresh initialization uses the selected finite-current profile
            # layout. Seeded restarts still preserve the donor's recorded
            # partition.
            finite_current_mode = extra_kwargs.get(
                "finite_current_mode",
                "wataru_proxy_field",
            )
            profile = module.get_finite_current_profile(finite_current_mode)
            num_vf_coils = 0
            if extra_kwargs.get("vf_template_path"):
                num_vf_coils = profile.default_num_vf_coils
            curves, proxy_coils, vf_coils = build_coil_bundle(
                num_proxy_coils=profile.default_num_proxy_coils,
                num_vf_coils=num_vf_coils,
            )
            fake_bs.coils = [
                *fake_tf_coils,
                *fake_banana_coils,
                *proxy_coils,
                *vf_coils,
            ]
            return (
                fake_bs,
                curves,
                fake_banana_curve,
                fake_banana_coils,
                proxy_coils,
                vf_coils,
                ("rc(0,1)", "zs(0,1)")
                if extra_kwargs.get("winding_surface_free_mpol", 0) > 0
                or extra_kwargs.get("winding_surface_free_ntor", 0) > 0
                else (),
            )

        def fake_curve_curve_distance(curves, *_args, **_kwargs):
            runtime["curve_curve_curves"] = tuple(curves)
            runtime["curve_curve_minimum_distances"].append(_args[0])
            return FakeCurveDistance()

        def fake_curve_surface_distance(
            curves, surface, minimum_distance, *_args, **_kwargs
        ):
            runtime["curve_surface_curves"] = tuple(curves)
            runtime["curve_surface_surface_label"] = surface.label
            return FakeCurveSurfaceDistance(curves, surface, minimum_distance)

        def fake_surface_surface_min_distance(surface_a, surface_b):
            runtime["surface_surface_min_distance_labels"] = (
                surface_a.label,
                surface_b.label,
            )
            self.assertIs(surface_a, fake_lcfs_surface)
            self.assertIs(surface_b, fake_vv)
            return 0.045

        def fake_projected_ellipse_width(*args, **_kwargs):
            runtime["projected_ellipse_width_args"] = args
            return FakeProjectedEllipseWidth()

        def fake_load_hardware_keepout(path, *, glb_path=None):
            runtime["hardware_keepout_loader_args"] = (path, glb_path)
            return (
                np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=float),
                0.0025,
                module.HARDWARE_KEEPOUT_MIN_DISTANCE_M,
                {"glb_sha256": "provenance-sha"},
            )

        def fake_hardware_keepout_metadata(path, glb_path=None):
            runtime["hardware_keepout_metadata_args"] = (path, glb_path)
            return {
                "HARDWARE_KEEPOUT_JSON": path,
                "HARDWARE_KEEPOUT_JSON_SHA256": "json-sha",
                "HARDWARE_KEEPOUT_GROUPS": ["shells", "limiter"],
                "HARDWARE_KEEPOUT_PROVENANCE_GLB": "provenance.glb",
                "HARDWARE_KEEPOUT_PROVENANCE_GLB_SHA256": "provenance-sha",
                "HARDWARE_KEEPOUT_LIVE_GLB": glb_path,
                "HARDWARE_KEEPOUT_LIVE_GLB_SHA256": "live-sha",
            }

        def fake_minimize(*_args, **_kwargs):
            runtime["minimize_calls"] += 1
            runtime["minimize_bounds"] = _kwargs.get("bounds")
            runtime["minimize_options"] = dict(_kwargs["options"])
            return SimpleNamespace(
                x=np.array([0.3, -0.2], dtype=float),
                nit=4,
                message="penalty_ok",
                success=True,
            )

        def fake_minimize_alm(*_args, **_kwargs):
            runtime["minimize_alm_calls"] += 1
            runtime["minimize_alm_options"] = dict(_args[4])
            accepted_callback = _kwargs.get("accepted_callback")
            if accepted_callback is not None and alm_accepted_candidate_x is not None:
                accepted_callback(np.asarray(alm_accepted_candidate_x, dtype=float))
            return SimpleNamespace(
                x=np.array([0.1, 0.2], dtype=float),
                nit=5,
                message="alm_ok",
                success=True,
                outer_iterations=2,
                penalty=3.5,
                multipliers=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float),
                constraint_values=np.array([0.0, 0.01, 0.0, 0.0, 0.0], dtype=float),
                solver_constraint_values=np.array(
                    [0.0, 0.2, 0.0, 0.0, 0.0], dtype=float
                ),
                normalized_constraint_values=np.array(
                    [0.0, 0.01, 0.0, 0.0, 0.0], dtype=float
                ),
                normalized_solver_constraint_values=np.array(
                    [0.0, 0.2, 0.0, 0.0, 0.0], dtype=float
                ),
                raw_constraint_values=np.array([0.0, 0.2, 0.0, 0.0, 0.0], dtype=float),
                raw_solver_constraint_values=np.array(
                    [0.0, 0.2, 0.0, 0.0, 0.0], dtype=float
                ),
                hard_signed_constraint_values=np.array(
                    [0.0, 0.02, 0.0, 0.0, 0.0], dtype=float
                ),
                hard_violation_values=np.array([0.0, 0.01, 0.0, 0.0, 0.0], dtype=float),
                surrogate_signed_constraint_values=np.array(
                    [0.0, 0.2, 0.0, 0.0, 0.0], dtype=float
                ),
                raw_hard_signed_constraint_values=np.array(
                    [0.0, 0.02, 0.0, 0.0, 0.0], dtype=float
                ),
                raw_hard_violation_values=np.array(
                    [0.0, 0.01, 0.0, 0.0, 0.0], dtype=float
                ),
                raw_surrogate_signed_constraint_values=np.array(
                    [0.0, 0.2, 0.0, 0.0, 0.0], dtype=float
                ),
                trust_radius=0.1,
                multiplier_cap_binding=True,
                multiplier_cap_binding_indices=[1],
                termination_reason="max_outer_after_subproblem_limit",
                converged_to_tolerances=False,
                restored_best_feasible=True,
                restored_best_feasible_reason="final_iterate_worse_than_best_feasible",
                optimizer_success=False,
                optimizer_message="STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
                final_max_feasibility_violation=0.01,
                final_stationarity_norm=0.02,
                final_raw_stationarity_norm=0.03,
                final_kkt_stationarity_norm=0.025,
                final_hard_max_violation=0.01,
                final_surrogate_max_value=0.2,
                hard_positive_shift_zero=True,
                signal_mismatch_active=False,
                final_penalty_gradient_norm=0.4,
                final_feasibility_tolerance=1.0e-3,
                final_stationarity_tolerance=5.0e-3,
                history=[{"outer_iteration": 1}],
            )

        def fake_capture_stage2_artifact_state(**kwargs):
            dofs = tuple(np.asarray(kwargs["dofs"], dtype=float).tolist())
            if artifact_state_by_x is None or dofs not in artifact_state_by_x:
                raise AssertionError(
                    f"unexpected artifact-state request for dofs {dofs}"
                )
            state = artifact_state_by_x[dofs]
            return {
                "x": np.asarray(kwargs["dofs"], dtype=float).copy(),
                "field_objective": float(state["field_objective"]),
                "coil_length": float(state["coil_length"]),
                "curve_curve_min_dist": float(state["curve_curve_min_dist"]),
                "curve_surface_min_dist": float(state["curve_surface_min_dist"]),
                "max_curvature": float(state["max_curvature"]),
                "poloidal_extent_rad": float(state["poloidal_extent_rad"]),
                "banana_current_A": float(state["banana_current_A"]),
                "tf_current_A": float(state["tf_current_A"]),
                "coil_width": float(state.get("coil_width", 0.10)),
                "width_min_threshold": float(state.get("width_min_threshold", 0.05)),
                "width_max_threshold": float(state.get("width_max_threshold", 0.17)),
                "self_intersect_penalty": float(
                    state.get("self_intersect_penalty", 0.0)
                ),
                "self_intersect_threshold": float(
                    state.get("self_intersect_threshold", 0.0)
                ),
                "shortest_self_distance": float(
                    state.get("shortest_self_distance", 0.5)
                ),
                "self_intersect_min_distance": float(
                    state.get("self_intersect_min_distance", 0.01)
                ),
                "self_envelope_mode": state.get("self_envelope_mode", "hinge"),
                "self_envelope_penalty": float(
                    state.get("self_envelope_penalty", 0.0)
                ),
                "self_envelope_min_dist": float(
                    state.get("self_envelope_min_dist", 0.08)
                ),
                "self_envelope_min_distance": float(
                    state.get("self_envelope_min_distance", 0.0462)
                ),
                "self_envelope_nominal_min_distance": state.get(
                    "self_envelope_nominal_min_distance"
                ),
                "self_envelope_sampling_margin": state.get(
                    "self_envelope_sampling_margin"
                ),
                "self_distance_window": float(
                    state.get("self_distance_window", 0.06)
                ),
                "self_envelope_groc_radius": state.get("self_envelope_groc_radius"),
                "self_envelope_groc_radius_floor": state.get(
                    "self_envelope_groc_radius_floor"
                ),
                "fold_penalty": float(state.get("fold_penalty", 0.0)),
                "fold_geodesic_curvature_max": float(
                    state.get("fold_geodesic_curvature_max", 30.0)
                ),
                "fold_geodesic_curvature_limit": float(
                    state.get("fold_geodesic_curvature_limit", 43.3114)
                ),
                "fold_geodesic_curvature_threshold": float(
                    state.get("fold_geodesic_curvature_threshold", 38.9803)
                ),
                "fold_ok": bool(state.get("fold_ok", True)),
                "hardware_status": {
                    "success": bool(state["hardware_status"]["success"]),
                    "violations": list(state["hardware_status"]["violations"]),
                },
                "stage2_iota_value": state.get("stage2_iota_value"),
                "stage2_iota_penalty": state.get("stage2_iota_penalty"),
                "stage2_iota_abs_error": state.get("stage2_iota_abs_error"),
                "stage2_iota_feasible": state.get("stage2_iota_feasible"),
                "stage2_iota_solve_failed": state.get("stage2_iota_solve_failed"),
            }

        def fake_run_basin_hopping(*_args, **_kwargs):
            runtime["run_basin_hopping_calls"] += 1
            self.assertEqual(_kwargs["basin_temperature"], 2.5)
            self.assertEqual(_kwargs["basin_niter_success"], 6)
            runtime["basin_bounds"] = _kwargs["minimizer_kwargs"].get("bounds")
            runtime["basin_options"] = dict(_kwargs["minimizer_kwargs"]["options"])
            return (
                SimpleNamespace(
                    x=np.array([0.6, -0.1], dtype=float),
                    fun=0.42,
                    nit=2,
                    minimization_failures=1,
                    lowest_optimization_result=SimpleNamespace(
                        nit=6,
                        message="basin_ok",
                        success=True,
                    ),
                ),
                self._EXPECTED_BASIN_TELEMETRY.copy(),
            )

        def fake_load_plasma_geometry(*args, **_kwargs):
            runtime["plasma_geometry_args"] = args
            return fake_plasma_geometry

        def fake_build_hbt_reference_surfaces(*_args, **kwargs):
            runtime["build_hbt_reference_surface_kwargs"] = dict(kwargs)
            return "hbt", "surf_coils", fake_vv

        def fake_probe_stage2_seed_bootability(**kwargs):
            runtime["stage2_iota_probes"] += 1
            runtime["stage2_iota_probe_kwargs"] = dict(kwargs)
            return {
                "BOOZER_BOOTABLE": True,
                "BOOZER_TRUSTED": True,
                "IOTA_NEAR_TARGET": True,
                "IOTA_FEASIBLE": True,
                "BOOTABILITY_REASON": "ok",
                "BOOTABILITY_STAGE": "post_gate_report",
                "BOOTABILITY_TARGET_IOTA": kwargs["iota_target"],
                "BOOTABILITY_SOLVED_IOTA": kwargs["iota_target"],
                "BOOTABILITY_SELF_INTERSECTING": False,
            }

        def fake_build_stage2_iota_runtime(**kwargs):
            runtime["stage2_iota_runtime_calls"] += 1
            runtime["stage2_iota_runtime_kwargs"] = dict(kwargs)
            return SimpleNamespace(
                mode=kwargs["mode"],
                boozer_surface=SimpleNamespace(),
                stats=SimpleNamespace(
                    bootstrap_seconds=0.25,
                    runtime_seconds=0.5,
                    runtime_calls=2,
                ),
                initial_state=SimpleNamespace(iota=0.18, penalty=0.0002),
                last_state=SimpleNamespace(
                    iota=0.201,
                    penalty=0.0,
                    abs_error=0.001,
                    feasible=True,
                    solve_failed=False,
                ),
                penalty_threshold=0.5 * 5.0e-3 * 5.0e-3,
                effective_weight=1.25,
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            equilibrium_path = Path(tmpdir) / "demo.nc"
            equilibrium_path.write_bytes(SIGNED_CW_WOUT_PATH.read_bytes())
            stage2_bs_path = str(Path(tmpdir) / "seed.json") if use_seed else None
            args = self._make_stage2_args(
                tmpdir,
                stage2_hardware_keepout_glb_default=(
                    module.DEFAULT_HARDWARE_KEEPOUT_GLB_PATH
                ),
                init_only=init_only,
                constraint_method=constraint_method,
                stage2_bs_path=stage2_bs_path,
                equilibrium_path=str(equilibrium_path),
                basin_hops=basin_hops,
                **dict(arg_overrides),
            )
            for attr_name in missing_attr_names:
                delattr(args, attr_name)

            with ExitStack() as stack:
                common_patches = [
                    patch.object(module, "validate_alm_cli_args", lambda *_args: None),
                    patch.object(
                        module,
                        "build_equilibrium_path",
                        lambda _args: args.equilibrium_path,
                    ),
                    patch.object(
                        module,
                        "create_equally_spaced_curves",
                        lambda *_args, **_kwargs: [FakeCurve() for _ in range(20)],
                    ),
                    patch.object(module, "Current", FakeCurrent),
                    patch.object(
                        module,
                        "Coil",
                        lambda curve, current: SimpleNamespace(
                            curve=curve, current=current
                        ),
                    ),
                    patch.object(
                        module,
                        "_load_stage2_vmec_surface",
                        lambda *_args, **_kwargs: fake_lcfs_surface,
                    ),
                    patch.object(
                        module,
                        "_load_plasma_geometry",
                        fake_load_plasma_geometry,
                    ),
                    patch.object(
                        module,
                        "build_hbt_reference_surfaces",
                        side_effect=fake_build_hbt_reference_surfaces,
                    ),
                    patch.object(
                        module,
                        "_surface_surface_min_distance",
                        side_effect=fake_surface_surface_min_distance,
                    ),
                    patch.object(
                        module,
                        "SquaredFlux",
                        lambda *_args, **_kwargs: FakeStage2Objective(0.5, [1.0, 1.0]),
                    ),
                    patch.object(
                        module,
                        "CurveLength",
                        lambda *_args, **_kwargs: FakeStage2Objective(1.6, [0.1, 0.2]),
                    ),
                    patch.object(
                        module,
                        "CurveCurveDistance",
                        side_effect=fake_curve_curve_distance,
                    ),
                    patch.object(
                        module,
                        "CurveSurfaceDistance",
                        side_effect=fake_curve_surface_distance,
                    ),
                    patch.object(
                        module,
                        "CurveVesselEnvelopeKeepout",
                        side_effect=FakeVesselEnvelopeKeepout,
                    ),
                    patch.object(
                        module,
                        "CurveVesselAvailableEnvelopeReward",
                        side_effect=FakeAvailableEnvelopeReward,
                    ),
                    patch.object(
                        module,
                        "CurveHardwareKeepout",
                        side_effect=FakeHardwareKeepout,
                    ),
                    patch.object(
                        module,
                        "load_hardware_keepout",
                        side_effect=fake_load_hardware_keepout,
                    ),
                    patch.object(
                        module,
                        "hardware_keepout_metadata",
                        side_effect=fake_hardware_keepout_metadata,
                    ),
                    patch.object(
                        module,
                        "LpCurveCurvature",
                        lambda *_args, **_kwargs: FakeCurvatureObjective(),
                    ),
                    patch.object(
                        module,
                        "PoloidalExtent",
                        lambda *_args, **_kwargs: FakeStage2Objective(0.0, [0.0, 0.0]),
                    ),
                    patch.object(
                        module,
                        "max_poloidal_extent_rad",
                        lambda *_args, **_kwargs: 0.0,
                    ),
                    patch.object(
                        module,
                        "ProjectedEllipseWidth",
                        side_effect=fake_projected_ellipse_width,
                    ),
                    patch.object(
                        module,
                        "CurveSelfIntersect",
                        lambda *_args, **_kwargs: FakeCurveSelfIntersect(),
                    ),
                    patch.object(
                        module,
                        "CurveSelfDistance",
                        lambda *_args, **_kwargs: FakeCurveSelfDistance(),
                    ),
                    patch.object(
                        module,
                        "CurveGlobalRadiusOfCurvature",
                        lambda *_args, **_kwargs: FakeCurveSelfDistance(),
                    ),
                    patch.object(
                        module,
                        "CurveSurfaceGeodesicCurvature",
                        lambda *_args, **_kwargs: FakeCurveSurfaceGeodesicCurvature(),
                    ),
                    patch.object(
                        module,
                        "FramedCurveSurfaceTangent",
                        lambda *_args, **_kwargs: SimpleNamespace(),
                    ),
                    patch.object(
                        module,
                        "QuadraticPenalty",
                        lambda *_args, **_kwargs: FakeStage2Objective(
                            0.05, [0.01, 0.02]
                        ),
                    ),
                    patch.object(
                        module,
                        "format_local_stage2_run_dir",
                        lambda *_args, **_kwargs: "runtime-smoke",
                    ),
                    patch.object(
                        module, "curves_to_vtk", lambda *_args, **_kwargs: None
                    ),
                    patch.object(
                        module, "cross_section_plot", lambda *_args, **_kwargs: None
                    ),
                    patch.object(
                        module, "_magnetic_field_plots", lambda *_args, **_kwargs: 0.03
                    ),
                    patch.object(
                        module, "is_self_intersecting", lambda *_args, **_kwargs: False
                    ),
                    patch.object(
                        module, "sha256_file", lambda *_args, **_kwargs: "0" * 64
                    ),
                    patch.object(
                        module,
                        "probe_stage2_seed_bootability",
                        side_effect=fake_probe_stage2_seed_bootability,
                    ),
                    patch.object(
                        module,
                        "build_stage2_iota_runtime",
                        side_effect=fake_build_stage2_iota_runtime,
                    ),
                    patch.object(
                        module,
                        "compute_stage2_bs_sha256",
                        return_value="0" * 64,
                    ),
                    patch.object(
                        module,
                        "save_boozer_surface_with_state",
                        side_effect=lambda _surface, path: path.with_suffix(
                            ".state.json"
                        ),
                    ),
                    patch.object(module, "minimize", side_effect=fake_minimize),
                    patch.object(module, "minimize_alm", side_effect=fake_minimize_alm),
                    patch.object(
                        module, "run_basin_hopping", side_effect=fake_run_basin_hopping
                    ),
                    patch.object(
                        module,
                        "write_json",
                        side_effect=lambda _path, data: runtime.__setitem__(
                            "results", data
                        ),
                    ),
                ]
                for patcher in common_patches:
                    stack.enter_context(patcher)
                if use_seed and seed_has_results_sidecar:
                    stack.enter_context(
                        patch.object(
                            module,
                            "load_stage2_seed_results",
                            return_value=(
                                Path(stage2_bs_path).with_name("results.json"),
                                seed_results_payload(),
                            ),
                        )
                    )
                if artifact_state_by_x is not None:
                    stack.enter_context(
                        patch.object(
                            module,
                            "_capture_stage2_artifact_state",
                            side_effect=fake_capture_stage2_artifact_state,
                        )
                    )
                if use_seed:
                    stack.enter_context(
                        patch.object(
                            module,
                            "load_stage2_seed_configuration",
                            side_effect=fake_seed_loader,
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            module,
                            "_initialize_coils",
                            side_effect=AssertionError(
                                "unexpected fresh initialization"
                            ),
                        )
                    )
                else:
                    stack.enter_context(
                        patch.object(
                            module,
                            "load_stage2_seed_configuration",
                            side_effect=AssertionError("unexpected seed load"),
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            module,
                            "_initialize_coils",
                            side_effect=fake_initialize_coils,
                        )
                    )
                module.main(args)

        return runtime

    def _assert_runtime_counts(
        self,
        runtime,
        *,
        seed_loads,
        initialize_calls,
        minimize_calls,
        minimize_alm_calls,
    ):
        self.assertEqual(runtime["seed_loads"], seed_loads)
        self.assertEqual(runtime["initialize_calls"], initialize_calls)
        self.assertEqual(runtime["minimize_calls"], minimize_calls)
        self.assertEqual(runtime["minimize_alm_calls"], minimize_alm_calls)
        expected_basin_calls = 1 if runtime["results"]["basin_hops"] > 0 else 0
        self.assertEqual(runtime["run_basin_hopping_calls"], expected_basin_calls)

    def _assert_banana_current_cap_rejected(self, runtime):
        self.assertFalse(runtime["results"]["OPTIMIZER_SUCCESS"])
        self.assertFalse(runtime["results"]["HARDWARE_CONSTRAINTS_OK"])
        self.assertTrue(
            any(
                "banana_current" in violation
                for violation in runtime["results"]["HARDWARE_CONSTRAINT_VIOLATIONS"]
            )
        )

    def _assert_init_only_runtime_counts(
        self, runtime, *, seed_loads, initialize_calls
    ):
        self._assert_runtime_counts(
            runtime,
            seed_loads=seed_loads,
            initialize_calls=initialize_calls,
            minimize_calls=0,
            minimize_alm_calls=0,
        )
        self.assertEqual(runtime["results"]["TERMINATION_MESSAGE"], "init_only")

    def test_stage2_main_init_only_loads_seed_and_writes_results(self):
        runtime = self._run_stage2_main(
            init_only=True, constraint_method="penalty", use_seed=True
        )

        self._assert_init_only_runtime_counts(
            runtime,
            seed_loads=1,
            initialize_calls=0,
        )
        self.assertTrue(runtime["results"]["OPTIMIZER_SUCCESS"])
        self.assertEqual(runtime["results"]["iterations"], 0)
        self.assertTrue(runtime["results"]["HARDWARE_CONSTRAINTS_OK"])
        self.assertTrue(runtime["results"]["STAGE2_BS_PATH"].endswith("seed.json"))

    def test_stage2_main_hardware_keepout_passes_glb_and_stamps_results(self):
        runtime = self._run_stage2_main(
            init_only=True,
            constraint_method="penalty",
            use_seed=True,
            arg_overrides={
                "stage2_vessel_keepout_weight": 1000.0,
                "stage2_hardware_keepout_weight": 1000.0,
                "stage2_hardware_keepout_json": "/tmp/hardware_keepout.json",
                "stage2_hardware_keepout_glb": "/tmp/hbt_assembly.glb",
            },
        )

        self._assert_init_only_runtime_counts(
            runtime,
            seed_loads=1,
            initialize_calls=0,
        )
        self.assertEqual(
            runtime["hardware_keepout_loader_args"],
            ("/tmp/hardware_keepout.json", "/tmp/hbt_assembly.glb"),
        )
        self.assertEqual(
            runtime["hardware_keepout_metadata_args"],
            ("/tmp/hardware_keepout.json", "/tmp/hbt_assembly.glb"),
        )
        self.assertIsNotNone(runtime["vessel_keepout_args"])
        hardware_args = runtime["hardware_keepout_objective_args"]
        self.assertIsNotNone(hardware_args)
        self.assertAlmostEqual(hardware_args["point_weight"], 0.0025)
        np.testing.assert_allclose(
            hardware_args["points"],
            np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=float),
        )

        results = runtime["results"]
        self.assertEqual(
            results["STAGE2_HARDWARE_KEEPOUT_GLB"],
            "/tmp/hbt_assembly.glb",
        )
        self.assertEqual(
            results["STAGE2_HARDWARE_KEEPOUT_JSON"],
            "/tmp/hardware_keepout.json",
        )
        self.assertEqual(
            results["EFFECTIVE_KEEPOUT_GROUPS"],
            ["vessel", "shells", "limiter"],
        )
        self.assertEqual(results["HARDWARE_KEEPOUT_JSON_SHA256"], "json-sha")
        self.assertEqual(
            results["HARDWARE_KEEPOUT_PROVENANCE_GLB_SHA256"],
            "provenance-sha",
        )
        self.assertEqual(results["HARDWARE_KEEPOUT_LIVE_GLB_SHA256"], "live-sha")
        self.assertAlmostEqual(results["STAGE2_HARDWARE_KEEPOUT_PENALTY"], 0.07)
        self.assertAlmostEqual(
            results["STAGE2_HARDWARE_KEEPOUT_MIN_DISTANCE_M"],
            0.042,
        )

    def test_stage2_main_rejects_self_envelope_hardware_status_violation(self):
        runtime = self._run_stage2_main(
            init_only=True,
            constraint_method="penalty",
            use_seed=True,
            self_envelope_min_dist=0.0460,
        )

        self.assertFalse(runtime["results"]["OPTIMIZER_SUCCESS"])
        self.assertFalse(runtime["results"]["HARDWARE_CONSTRAINTS_OK"])
        self.assertIn(
            "self_envelope_min_dist 0.046000 below threshold 0.046200",
            runtime["results"]["HARDWARE_CONSTRAINT_VIOLATIONS"],
        )
        self.assertAlmostEqual(
            runtime["results"]["SELF_ENVELOPE_MIN_DIST_M"],
            0.0460,
        )
        self.assertAlmostEqual(
            runtime["results"]["SELF_ENVELOPE_THRESHOLD_M"],
            0.0462,
        )

    def test_stage2_main_sampling_margin_tightens_self_envelope_gate(self):
        runtime = self._run_stage2_main(
            init_only=True,
            constraint_method="penalty",
            use_seed=True,
            self_envelope_min_dist=0.0480,
            arg_overrides={"self_envelope_sampling_margin": 0.0030},
        )

        self.assertFalse(runtime["results"]["OPTIMIZER_SUCCESS"])
        self.assertFalse(runtime["results"]["HARDWARE_CONSTRAINTS_OK"])
        self.assertIn(
            "self_envelope_min_dist 0.048000 below threshold 0.049200",
            runtime["results"]["HARDWARE_CONSTRAINT_VIOLATIONS"],
        )
        self.assertAlmostEqual(
            runtime["results"]["SELF_ENVELOPE_THRESHOLD_M"],
            0.0492,
        )
        self.assertAlmostEqual(
            runtime["results"]["SELF_ENVELOPE_NOMINAL_MIN_DISTANCE_M"],
            0.0462,
        )
        self.assertAlmostEqual(
            runtime["results"]["SELF_ENVELOPE_SAMPLING_MARGIN_M"],
            0.0030,
        )

    def test_stage2_main_report_runs_post_gate_without_hot_loop_runtime(self):
        runtime = self._run_stage2_main(
            init_only=True,
            constraint_method="penalty",
            use_seed=True,
            arg_overrides={
                "stage2_iota_target": 0.2,
            },
        )

        self.assertEqual(runtime["stage2_iota_probes"], 1)
        self.assertTrue(runtime["results"]["STAGE2_ROOT_FIX_ENABLED"])
        self.assertFalse(runtime["results"]["STAGE2_IOTA_OBJECTIVE_COUPLED"])
        self.assertFalse(runtime["results"]["STAGE2_IOTA_HOT_LOOP_ENABLED"])
        self.assertTrue(runtime["results"]["BOOZER_BOOTABLE"])
        self.assertTrue(runtime["results"]["BOOZER_TRUSTED"])
        self.assertEqual(runtime["stage2_iota_probe_kwargs"]["iota_target"], 0.2)

    def test_stage2_main_soft_iota_mode_builds_active_runtime(self):
        runtime = self._run_stage2_main(
            init_only=False,
            constraint_method="penalty",
            use_seed=True,
            artifact_state_by_x={
                (0.3, -0.2): {
                    "field_objective": 0.2,
                    "coil_length": 1.6,
                    "curve_curve_min_dist": 0.08,
                    "curve_surface_min_dist": 0.04,
                    "max_curvature": 20.0,
                    "poloidal_extent_rad": 0.1,
                    "banana_current_A": 9500.0,
                    "tf_current_A": -80000.0,
                    "hardware_status": {"success": True, "violations": []},
                    "stage2_iota_value": 0.201,
                    "stage2_iota_penalty": 0.0,
                    "stage2_iota_abs_error": 0.001,
                    "stage2_iota_feasible": True,
                    "stage2_iota_solve_failed": False,
                }
            },
            arg_overrides={
                "stage2_iota_target": 0.2,
                "stage2_iota_objective_mode": "soft",
            },
        )

        self.assertEqual(runtime["stage2_iota_runtime_calls"], 1)
        self.assertEqual(runtime["stage2_iota_runtime_kwargs"]["mode"], "soft")
        self.assertEqual(runtime["stage2_iota_runtime_kwargs"]["iota_target"], 0.2)
        self.assertEqual(runtime["minimize_calls"], 1)
        self.assertTrue(runtime["results"]["STAGE2_IOTA_OBJECTIVE_COUPLED"])
        self.assertTrue(runtime["results"]["STAGE2_IOTA_HOT_LOOP_ENABLED"])
        self.assertEqual(runtime["results"]["STAGE2_IOTA_OBJECTIVE_MODE"], "soft")
        self.assertEqual(runtime["results"]["STAGE2_IOTA_VALUE"], 0.201)

    def test_stage2_main_reports_lcfs_metrics_and_boundary_clearance(self):
        module = load_stage2_module()
        runtime = self._run_stage2_main(
            init_only=True,
            constraint_method="penalty",
            use_seed=False,
        )

        self._assert_init_only_runtime_counts(
            runtime,
            seed_loads=0,
            initialize_calls=1,
        )
        self.assertEqual(
            runtime["surface_surface_min_distance_labels"],
            ("lcfs", "vv"),
        )
        self.assertEqual(
            runtime["plasma_geometry_args"][0],
            in_bounds_lcfs_major_radius_m(),
        )
        self.assertEqual(runtime["results"]["MAJOR_RADIUS"], 0.976)
        self.assertEqual(
            runtime["results"]["BANANA_WINDING_SURFACE_MAJOR_RADIUS_M"],
            module.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
        )
        self.assertEqual(
            runtime["results"]["COIL_WINDING_SURFACE_MAJOR_RADIUS_M"],
            module.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
        )
        self.assertEqual(
            runtime["results"]["FINAL_LCFS_MAJOR_RADIUS_M"],
            in_bounds_lcfs_major_radius_m(),
        )
        self.assertEqual(
            runtime["results"]["FINAL_LCFS_MINOR_RADIUS_M"],
            in_bounds_lcfs_minor_radius_m(),
        )
        self.assertNotEqual(
            runtime["results"]["MAJOR_RADIUS"],
            runtime["results"]["FINAL_LCFS_MAJOR_RADIUS_M"],
        )
        self.assertNotEqual(runtime["results"]["FINAL_LCFS_MAJOR_RADIUS_M"], 0.88)
        self.assertNotEqual(runtime["results"]["FINAL_LCFS_MINOR_RADIUS_M"], 0.12)
        self.assertEqual(runtime["curve_surface_surface_label"], "lcfs")
        # The hard coil-coil gate stays at --cc-threshold; the optimizer is
        # additionally steered toward the buffered objective threshold
        # (CC_THRESHOLD + BANANA_CC_OBJECTIVE_MARGIN_M).
        self.assertEqual(
            runtime["curve_curve_minimum_distances"],
            [0.05, 0.05 + module.BANANA_CC_OBJECTIVE_MARGIN_M],
        )
        self.assertEqual(
            runtime["projected_ellipse_width_args"][1],
            module.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
        )
        self.assertEqual(runtime["projected_ellipse_width_args"][2], 0.21)

    def test_stage2_main_rejects_wataru_seed_without_results_sidecar(self):
        workflow_runner_common = load_workflow_runner_common_module()

        with self.assertRaisesRegex(
            ValueError,
            re.escape(workflow_runner_common.STAGE2_SIDECAR_REQUIRED_ERROR),
        ):
            self._run_stage2_main(
                init_only=True,
                constraint_method="penalty",
                use_seed=True,
                seed_has_results_sidecar=False,
                arg_overrides={"finite_current_mode": "wataru_proxy_field"},
            )

    def test_stage2_main_init_only_wataru_proxy_field_uses_repo_default_vf_and_banana_only_penalties(
        self,
    ):
        workflow_helpers = load_workflow_helpers_module()
        vf_current_A = 9000.0 / 6.5
        runtime = self._run_stage2_main(
            init_only=True,
            constraint_method="penalty",
            use_seed=False,
            arg_overrides={
                "finite_current_mode": "wataru_proxy_field",
                "proxy_plasma_current_A": 9000.0,
                "vf_current_A": vf_current_A,
            },
        )

        self._assert_init_only_runtime_counts(
            runtime,
            seed_loads=0,
            initialize_calls=1,
        )
        self.assertEqual(
            runtime["results"]["FINITE_CURRENT_MODE"], "wataru_proxy_field"
        )
        self.assertEqual(runtime["results"]["NUM_PROXY_COILS"], 1)
        self.assertEqual(runtime["results"]["NUM_VF_COILS"], 20)
        self.assertEqual(
            runtime["results"]["PROXY_VF_CURRENT_SCALAR_POLICY"],
            "nonnegative_magnitude",
        )
        self.assertEqual(runtime["results"]["PROXY_PLASMA_CURRENT_A"], 9000.0)
        self.assertEqual(runtime["results"]["VF_CURRENT_A"], vf_current_A)
        self.assertEqual(runtime["results"]["VF_CURRENT_MAX_A"], 1.6e4)
        self.assertEqual(
            runtime["results"]["VF_TEMPLATE_PATH"],
            workflow_helpers.default_wataru_vf_template_path(),
        )
        self.assertEqual(len(runtime["curve_curve_curves"]), 1)
        self.assertEqual(len(runtime["curve_surface_curves"]), 1)
        self.assertEqual(
            runtime["curve_surface_curves"][0],
            runtime["curve_curve_curves"][0],
        )
        self.assertEqual(
            runtime["initialize_extra_kwargs"]["vf_template_path"],
            workflow_helpers.default_wataru_vf_template_path(),
        )
        self.assertEqual(
            runtime["initialize_extra_kwargs"]["finite_current_mode"],
            "wataru_proxy_field",
        )
        self.assertEqual(
            runtime["initialize_extra_kwargs"]["surface_scale_factor"], 0.75
        )

    def test_stage2_main_init_only_vacuum_uses_tf_banana_only_field(self):
        runtime = self._run_stage2_main(
            init_only=True,
            constraint_method="penalty",
            use_seed=False,
            arg_overrides={"finite_current_mode": "vacuum"},
        )

        self._assert_init_only_runtime_counts(
            runtime,
            seed_loads=0,
            initialize_calls=1,
        )
        self.assertEqual(runtime["results"]["FINITE_CURRENT_MODE"], "vacuum")
        self.assertEqual(runtime["results"]["NUM_PROXY_COILS"], 0)
        self.assertEqual(runtime["results"]["NUM_VF_COILS"], 0)
        self.assertEqual(runtime["results"]["TOTAL_COILS"], 21)
        self.assertEqual(runtime["results"]["PROXY_PLASMA_CURRENT_A"], 0.0)
        self.assertEqual(runtime["results"]["VF_CURRENT_A"], 0.0)
        self.assertIsNone(runtime["results"]["VF_TEMPLATE_PATH"])
        self.assertEqual(runtime["results"]["PROXY_PLACEMENT_MODE"], "none")
        self.assertEqual(runtime["results"]["PROXY_VF_CURRENT_SCALAR_POLICY"], "none")
        self.assertEqual(runtime["results"]["VF_CURRENT_SIGN_POLICY"], "none")
        self.assertEqual(runtime["results"]["VF_CURRENT_MUTABILITY"], "none")
        self.assertEqual(len(runtime["curve_curve_curves"]), 1)
        self.assertEqual(len(runtime["curve_surface_curves"]), 1)
        self.assertIsNone(runtime["initialize_extra_kwargs"]["vf_template_path"])
        self.assertEqual(
            runtime["initialize_extra_kwargs"]["finite_current_mode"],
            "vacuum",
        )

    def test_stage2_main_wataru_mode_ignores_jhalpern_banana_pin_env(self):
        with patch.dict(os.environ, {"BANANA_I_FIXED_S2": "not-a-number"}):
            runtime = self._run_stage2_main(
                init_only=True,
                constraint_method="penalty",
                use_seed=False,
                arg_overrides={"finite_current_mode": "wataru_proxy_field"},
            )

        self.assertEqual(
            runtime["results"]["FINITE_CURRENT_MODE"], "wataru_proxy_field"
        )
        self.assertFalse(runtime["results"]["BANANA_CURRENT_PINNED"])
        self.assertIsNone(runtime["results"]["BANANA_I_FIXED_S2_KA"])

    def test_stage2_main_rejects_wataru_proxy_vf_ratio_drift(self):
        with self.assertRaisesRegex(ValueError, "proxy/VF convention"):
            self._run_stage2_main(
                init_only=True,
                constraint_method="penalty",
                use_seed=False,
                arg_overrides={
                    "finite_current_mode": "wataru_proxy_field",
                    "proxy_plasma_current_A": 9000.0,
                    "vf_current_A": 500.0,
                },
            )

    def test_stage2_main_init_only_wataru_seed_restart_uses_banana_only_penalties(self):
        vf_current_A = 9000.0 / 6.5
        runtime = self._run_stage2_main(
            init_only=True,
            constraint_method="penalty",
            use_seed=True,
            arg_overrides={"finite_current_mode": "wataru_proxy_field"},
            seed_stage2_results={
                "PLASMA_SURF_FILENAME": "demo.nc",
                "TF_CURRENT_A": -8.0e4,
                "NUM_TF_COILS": 20,
                "NUM_BANANA_COILS": 1,
                "NUM_PROXY_COILS": 1,
                "NUM_VF_COILS": 1,
                "FINITE_CURRENT_MODE": "wataru_proxy_field",
                "BOOZER_CURRENT_CONVENTION": "mu0",
                "PROXY_PLASMA_CURRENT_A": 9000.0,
                "VF_CURRENT_A": vf_current_A,
                "VF_TEMPLATE_PATH": "/tmp/vf_template.json",
            },
        )

        self._assert_init_only_runtime_counts(
            runtime,
            seed_loads=1,
            initialize_calls=0,
        )
        self.assertEqual(
            runtime["results"]["FINITE_CURRENT_MODE"], "wataru_proxy_field"
        )
        self.assertEqual(runtime["results"]["NUM_PROXY_COILS"], 1)
        self.assertEqual(runtime["results"]["NUM_VF_COILS"], 1)
        self.assertEqual(len(runtime["curve_curve_curves"]), 1)
        self.assertEqual(len(runtime["curve_surface_curves"]), 1)
        self.assertEqual(
            runtime["curve_surface_curves"][0],
            runtime["curve_curve_curves"][0],
        )

    def test_stage2_main_init_only_legacy_zero_vf_seed_restart_keeps_donor_partition(
        self,
    ):
        runtime = self._run_stage2_main(
            init_only=True,
            constraint_method="penalty",
            use_seed=True,
            arg_overrides={"finite_current_mode": "wataru_proxy_field"},
            seed_stage2_results={
                "PLASMA_SURF_FILENAME": "demo.nc",
                "TF_CURRENT_A": -8.0e4,
                "NUM_TF_COILS": 20,
                "NUM_BANANA_COILS": 1,
                "NUM_PROXY_COILS": 1,
                "NUM_VF_COILS": 0,
                "FINITE_CURRENT_MODE": "wataru_proxy_field",
                "BOOZER_CURRENT_CONVENTION": "mu0",
                "PROXY_PLASMA_CURRENT_A": 0.0,
                "VF_CURRENT_A": 0.0,
                "VF_TEMPLATE_PATH": None,
            },
        )

        self._assert_init_only_runtime_counts(
            runtime,
            seed_loads=1,
            initialize_calls=0,
        )
        self.assertEqual(
            runtime["results"]["FINITE_CURRENT_MODE"], "wataru_proxy_field"
        )
        self.assertEqual(runtime["results"]["NUM_PROXY_COILS"], 1)
        self.assertEqual(runtime["results"]["NUM_VF_COILS"], 0)
        self.assertEqual(runtime["results"]["PROXY_PLASMA_CURRENT_A"], 0.0)
        self.assertEqual(runtime["results"]["VF_CURRENT_A"], 0.0)
        self.assertIsNone(runtime["results"]["VF_TEMPLATE_PATH"])
        self.assertEqual(len(runtime["curve_curve_curves"]), 1)
        self.assertEqual(len(runtime["curve_surface_curves"]), 1)
        self.assertEqual(
            runtime["curve_surface_curves"][0],
            runtime["curve_curve_curves"][0],
        )

    def test_stage2_main_alm_path_uses_minimize_alm(self):
        runtime = self._run_stage2_main(
            init_only=False, constraint_method="alm", use_seed=True
        )

        self._assert_runtime_counts(
            runtime,
            seed_loads=1,
            initialize_calls=0,
            minimize_calls=0,
            minimize_alm_calls=1,
        )
        self.assertEqual(runtime["results"]["CONSTRAINT_METHOD"], "alm")
        self.assertEqual(runtime["results"]["ALM_OUTER_ITERATIONS"], 2)
        self.assertEqual(
            runtime["results"]["ALM_TERMINATION_REASON"],
            "max_outer_after_subproblem_limit",
        )
        self.assertFalse(runtime["results"]["ALM_CONVERGED"])
        self.assertTrue(runtime["results"]["ALM_RESTORED_BEST_FEASIBLE"])
        self.assertEqual(
            runtime["results"]["ALM_RESTORED_BEST_FEASIBLE_REASON"],
            "final_iterate_worse_than_best_feasible",
        )
        self.assertFalse(runtime["results"]["ALM_INNER_OPTIMIZER_SUCCESS"])
        self.assertEqual(
            runtime["results"]["ALM_INNER_OPTIMIZER_MESSAGE"],
            "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
        )
        self.assertEqual(
            runtime["results"]["ALM_FINAL_MAX_FEASIBILITY_VIOLATION"],
            0.01,
        )
        self.assertEqual(runtime["results"]["ALM_FINAL_STATIONARITY_NORM"], 0.02)
        self.assertEqual(runtime["results"]["ALM_FINAL_RAW_STATIONARITY_NORM"], 0.03)
        self.assertEqual(runtime["results"]["ALM_FINAL_KKT_STATIONARITY_NORM"], 0.025)
        np.testing.assert_allclose(
            runtime["results"]["ALM_FINAL_HARD_SIGNED_CONSTRAINT_VALUES"],
            [0.0, 0.02, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            runtime["results"]["ALM_FINAL_HARD_VIOLATION_VALUES"],
            [0.0, 0.01, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            runtime["results"]["ALM_FINAL_SURROGATE_SIGNED_CONSTRAINT_VALUES"],
            [0.0, 0.2, 0.0, 0.0, 0.0],
        )
        self.assertEqual(runtime["results"]["ALM_FINAL_HARD_MAX_VIOLATION"], 0.01)
        self.assertEqual(runtime["results"]["ALM_FINAL_SURROGATE_MAX_VALUE"], 0.2)
        self.assertTrue(runtime["results"]["ALM_FINAL_HARD_POSITIVE_SHIFT_ZERO"])
        self.assertFalse(runtime["results"]["ALM_FINAL_SIGNAL_MISMATCH_ACTIVE"])
        self.assertEqual(runtime["results"]["ALM_FINAL_PENALTY_GRADIENT_NORM"], 0.4)
        self.assertEqual(runtime["results"]["ALM_FINAL_FEASIBILITY_TOL"], 1.0e-3)
        self.assertEqual(runtime["results"]["ALM_FINAL_STATIONARITY_TOL"], 5.0e-3)
        self.assertTrue(runtime["results"]["ALM_MULTIPLIER_CAP_BINDING"])
        self.assertEqual(runtime["results"]["ALM_MULTIPLIER_CAP_BINDING_INDICES"], [1])
        self.assertEqual(runtime["results"]["TERMINATION_MESSAGE"], "alm_ok")
        self.assertEqual(
            runtime["minimize_alm_options"]["maxcor"],
            runtime["default_maxcor"],
        )

    def test_stage2_main_alm_restores_best_exact_hardware_pass_for_artifact_output(
        self,
    ):
        def make_artifact_state(
            field_objective,
            coil_length,
            *,
            success,
            curve_curve_min_dist=0.06,
            max_curvature=41.0,
        ):
            violations = (
                [] if success else [f"coil_length {coil_length:.6f} > 2.000000"]
            )
            return {
                "field_objective": float(field_objective),
                "coil_length": float(coil_length),
                "curve_curve_min_dist": float(curve_curve_min_dist),
                "curve_surface_min_dist": 0.02,
                "max_curvature": float(max_curvature),
                "poloidal_extent_rad": 0.0,
                "banana_current_A": 9500.0,
                "tf_current_A": -8.0e4,
                "hardware_status": {
                    "success": bool(success),
                    "violations": violations,
                },
            }

        runtime = self._run_stage2_main(
            init_only=False,
            constraint_method="alm",
            use_seed=True,
            alm_accepted_candidate_x=np.array([0.9, 0.8], dtype=float),
            artifact_state_by_x={
                (0.0, 0.0): make_artifact_state(0.9, 2.0004, success=False),
                (0.9, 0.8): make_artifact_state(
                    0.4,
                    1.69,
                    success=True,
                    curve_curve_min_dist=0.07,
                    max_curvature=39.0,
                ),
                (0.1, 0.2): make_artifact_state(0.6, 2.0002, success=False),
            },
        )

        self._assert_runtime_counts(
            runtime,
            seed_loads=1,
            initialize_calls=0,
            minimize_calls=0,
            minimize_alm_calls=1,
        )
        self.assertEqual(
            runtime["results"]["TERMINATION_MESSAGE"],
            "alm_ok; restored_best_exact_hardware_pass",
        )
        self.assertFalse(runtime["results"]["OPTIMIZER_SUCCESS"])
        self.assertTrue(runtime["results"]["HARDWARE_CONSTRAINTS_OK"])
        self.assertEqual(runtime["results"]["HARDWARE_CONSTRAINT_VIOLATIONS"], [])
        self.assertEqual(runtime["results"]["COIL_LENGTH"], 1.69)

    def test_stage2_main_penalty_path_uses_lbfgsb(self):
        runtime = self._run_stage2_main(
            init_only=False, constraint_method="penalty", use_seed=True
        )

        self._assert_runtime_counts(
            runtime,
            seed_loads=1,
            initialize_calls=0,
            minimize_calls=1,
            minimize_alm_calls=0,
        )
        self.assertEqual(runtime["results"]["CONSTRAINT_METHOD"], "penalty")
        self.assertEqual(runtime["results"]["TERMINATION_MESSAGE"], "penalty_ok")
        self.assertEqual(runtime["results"]["BANANA_INIT_CURRENT_A"], 9500.0)
        self.assertEqual(runtime["results"]["BANANA_CURRENT_MAX_A"], 1.6e4)
        self.assertIsNotNone(runtime["minimize_bounds"])
        self.assertEqual(
            runtime["minimize_options"]["maxcor"], runtime["default_maxcor"]
        )

    def test_stage2_main_basin_hopping_persists_telemetry(self):
        runtime = self._run_stage2_main(
            init_only=False,
            constraint_method="penalty",
            use_seed=True,
            basin_hops=2,
        )

        self._assert_runtime_counts(
            runtime,
            seed_loads=1,
            initialize_calls=0,
            minimize_calls=0,
            minimize_alm_calls=0,
        )
        self.assertEqual(runtime["results"]["basin_hops"], 2)
        self.assertEqual(runtime["results"]["TERMINATION_MESSAGE"], "basin_ok")
        self.assertEqual(runtime["results"]["basin_iterations"], 2)
        self.assertEqual(runtime["results"]["basin_minimization_failures"], 1)
        self.assertEqual(runtime["results"]["basin_temperature"], 2.5)
        self.assertEqual(runtime["results"]["basin_niter_success"], 6)
        self.assertEqual(runtime["results"]["basin_completed_hops"], 2)
        self.assertEqual(runtime["results"]["basin_initial_objective"], 0.55)
        self.assertEqual(runtime["results"]["basin_best_hop_objective"], 0.42)
        self.assertEqual(runtime["results"]["basin_best_hop_index"], 1)
        self.assertEqual(runtime["results"]["basin_best_result_source"], "hop")
        self.assertEqual(runtime["results"]["basin_objective_improvement"], 0.13)
        self.assertEqual(runtime["results"]["basin_nonfinite_rejections"], 0)
        self.assertEqual(runtime["results"]["basin_normalized_step_rejections"], 1)
        self.assertIsNotNone(runtime["basin_bounds"])
        self.assertEqual(runtime["basin_options"]["maxcor"], runtime["default_maxcor"])

    def test_stage2_main_rejects_final_banana_current_above_cap(self):
        runtime = self._run_stage2_main(
            init_only=True,
            constraint_method="penalty",
            use_seed=False,
            banana_current_A=17000.0,
        )

        self._assert_banana_current_cap_rejected(runtime)

    def test_stage2_main_rejects_negative_final_banana_current_above_cap_magnitude(
        self,
    ):
        runtime = self._run_stage2_main(
            init_only=True,
            constraint_method="penalty",
            use_seed=False,
            banana_current_A=-17000.0,
        )

        self._assert_banana_current_cap_rejected(runtime)

    def test_stage2_main_records_loaded_seed_current_as_initial_current(self):
        runtime = self._run_stage2_main(
            init_only=True,
            constraint_method="penalty",
            use_seed=True,
            banana_current_A=12345.0,
        )

        self.assertEqual(runtime["results"]["BANANA_INIT_CURRENT_A"], 12345.0)

    def test_stage2_main_fresh_init_path_uses_initialize_coils(self):
        runtime = self._run_stage2_main(
            init_only=True, constraint_method="penalty", use_seed=False
        )

        self._assert_init_only_runtime_counts(
            runtime,
            seed_loads=0,
            initialize_calls=1,
        )
        self.assertIsNone(runtime["results"]["STAGE2_BS_PATH"])

    def test_stage2_main_fresh_init_forwards_winding_surface_shape_modes(self):
        runtime = self._run_stage2_main(
            init_only=True,
            constraint_method="penalty",
            use_seed=False,
            arg_overrides=(
                ("winding_surface_free_mpol", 1),
                ("winding_surface_free_ntor", 1),
            ),
        )

        self.assertEqual(
            runtime["build_hbt_reference_surface_kwargs"][
                "winding_surface_free_mpol"
            ],
            1,
        )
        self.assertEqual(
            runtime["build_hbt_reference_surface_kwargs"][
                "winding_surface_free_ntor"
            ],
            1,
        )
        self.assertEqual(
            runtime["initialize_extra_kwargs"]["winding_surface_free_mpol"],
            1,
        )
        self.assertEqual(
            runtime["initialize_extra_kwargs"]["winding_surface_free_ntor"],
            1,
        )

    def test_configure_winding_surface_shape_dofs_unfixes_low_modes(self):
        from simsopt.geo import SurfaceRZFourier
        from banana_opt.stage2_geometry import configure_winding_surface_shape_dofs

        surf = SurfaceRZFourier(nfp=2, stellsym=True, mpol=2, ntor=1)
        surf.set_rc(0, 0, 0.903)
        surf.set_rc(1, 0, 0.142)
        surf.set_zs(1, 0, 0.142)

        free_names = configure_winding_surface_shape_dofs(
            surf,
            free_mpol=1,
            free_ntor=1,
        )

        self.assertGreater(len(free_names), 0)
        self.assertTrue(surf.is_fixed("rc(0,0)"))
        self.assertTrue(surf.is_fixed("rc(1,0)"))
        self.assertTrue(surf.is_fixed("zs(1,0)"))
        self.assertTrue(any(name in free_names for name in ("rc(0,1)", "zs(0,1)")))


class AlmUtilsTests(unittest.TestCase):
    def test_validate_resume_alm_state_requires_matching_constraint_names(self):
        module = load_single_stage_example_module()
        resume_state = {
            "constraint_names": ["coil_surface_spacing"],
            "constraint_scales": [0.05],
            "multipliers": [2.0],
            "penalty": 3.0,
        }

        multipliers, penalty = module.validate_resume_alm_state(
            resume_state,
            ["coil_surface_spacing"],
            [0.05],
        )

        np.testing.assert_allclose(multipliers, [2.0])
        self.assertEqual(penalty, 3.0)
        with self.assertRaisesRegex(ValueError, "constraint_names mismatch"):
            module.validate_resume_alm_state(
                resume_state,
                ["coil_coil_spacing"],
                [0.05],
            )

    def test_validate_resume_alm_state_rejects_multiplier_length_mismatch(self):
        module = load_single_stage_example_module()

        with self.assertRaisesRegex(ValueError, "multiplier length mismatch"):
            module.validate_resume_alm_state(
                {
                    "constraint_names": ["coil_surface_spacing", "max_curvature"],
                    "constraint_scales": [0.05, 100.0],
                    "multipliers": [2.0],
                    "penalty": 3.0,
                },
                ["coil_surface_spacing", "max_curvature"],
                [0.05, 100.0],
            )

    def test_validate_resume_alm_state_rejects_legacy_state_without_names_or_scales(
        self,
    ):
        module = load_single_stage_example_module()

        with self.assertRaisesRegex(ValueError, "constraint_names"):
            module.validate_resume_alm_state(
                {
                    "multipliers": [2.0],
                    "penalty": 3.0,
                },
                ["coil_surface_spacing"],
                [0.05],
            )
        with self.assertRaisesRegex(ValueError, "constraint_scales"):
            module.validate_resume_alm_state(
                {
                    "constraint_names": ["coil_surface_spacing"],
                    "multipliers": [2.0],
                    "penalty": 3.0,
                },
                ["coil_surface_spacing"],
                [0.05],
            )

    def test_validate_resume_alm_state_rejects_scale_mismatch(self):
        module = load_single_stage_example_module()

        with self.assertRaisesRegex(ValueError, "constraint_scales mismatch"):
            module.validate_resume_alm_state(
                {
                    "constraint_names": ["coil_surface_spacing"],
                    "constraint_scales": [0.05],
                    "multipliers": [2.0],
                    "penalty": 3.0,
                },
                ["coil_surface_spacing"],
                [0.10],
            )

    def test_validate_resume_alm_state_rejects_nan_multipliers(self):
        module = load_single_stage_example_module()

        with self.assertRaisesRegex(
            ValueError, r"ALM initial_multipliers non-finite at indices \[1\]"
        ):
            module.validate_resume_alm_state(
                {
                    "constraint_names": ["coil_surface_spacing", "max_curvature"],
                    "constraint_scales": [0.05, 100.0],
                    "multipliers": [2.0, float("nan")],
                    "penalty": 3.0,
                },
                ["coil_surface_spacing", "max_curvature"],
                [0.05, 100.0],
            )

    def test_validate_resume_alm_state_rejects_negative_multipliers(self):
        module = load_single_stage_example_module()

        with self.assertRaisesRegex(
            ValueError, r"ALM initial_multipliers negative at indices \[0\]"
        ):
            module.validate_resume_alm_state(
                {
                    "constraint_names": ["coil_surface_spacing", "max_curvature"],
                    "constraint_scales": [0.05, 100.0],
                    "multipliers": [-1.0, 2.0],
                    "penalty": 3.0,
                },
                ["coil_surface_spacing", "max_curvature"],
                [0.05, 100.0],
            )

    def test_validate_resume_alm_state_rejects_inf_multipliers(self):
        module = load_single_stage_example_module()

        with self.assertRaisesRegex(
            ValueError, r"ALM initial_multipliers non-finite at indices \[0\]"
        ):
            module.validate_resume_alm_state(
                {
                    "constraint_names": ["coil_surface_spacing"],
                    "constraint_scales": [0.05],
                    "multipliers": [float("inf")],
                    "penalty": 3.0,
                },
                ["coil_surface_spacing"],
                [0.05],
            )

    def test_current_solver_checkpoint_alm_state_includes_constraint_names(self):
        module = load_single_stage_example_module()
        module.PRESERVED_TIMEOUT_REPLAY_CONFIG = module.replace(
            module.PRESERVED_TIMEOUT_REPLAY_CONFIG,
            constraint_method="alm",
        )

        module.set_alm_runtime_state([1.0, 2.0], 3.0, ["gap", "current"])
        module.run_dict = {
            "search_eval": {"constraint_scales": np.array([0.05, 16000.0])}
        }
        alm_state = module.current_solver_checkpoint_alm_state()

        self.assertEqual(alm_state["constraint_names"], ["gap", "current"])
        np.testing.assert_allclose(alm_state["constraint_scales"], [0.05, 16000.0])
        np.testing.assert_allclose(alm_state["multipliers"], [1.0, 2.0])
        self.assertEqual(alm_state["penalty"], 3.0)

    def test_upper_bound_residual_clamps_negative_values(self):
        module = load_alm_utils_module()

        self.assertEqual(module.upper_bound_residual(1.0, 2.0), 0.0)
        self.assertEqual(module.upper_bound_residual(2.5, 2.0), 0.5)

    def test_augmented_inequality_objective_combines_base_and_constraints(self):
        module = load_alm_utils_module()

        evaluation = module.augmented_inequality_objective(
            base_value=3.0,
            base_grad=np.array([1.0, -1.0]),
            constraint_values=[0.5, 0.0],
            constraint_grads=[np.array([2.0, 0.0]), np.array([0.0, 0.0])],
            multipliers=np.array([1.0, 7.0]),
            penalty=10.0,
        )

        self.assertAlmostEqual(evaluation["total"], 4.75)
        np.testing.assert_allclose(evaluation["grad"], np.array([13.0, -1.0]))
        self.assertAlmostEqual(evaluation["max_violation"], 0.5)

    def test_minimize_alm_solves_simple_quadratic_with_signed_upper_bound_constraint(
        self,
    ):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=6,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
        )

        def evaluate_problem(x, multipliers, penalty):
            value = 0.5 * (x[0] - 2.0) ** 2
            grad = np.array([x[0] - 2.0])
            signed_constraint_value = np.array([x[0] - 1.0])
            constraint_grad = [np.array([1.0])]
            return module.augmented_inequality_objective(
                value,
                grad,
                signed_constraint_value,
                constraint_grad,
                multipliers,
                penalty,
            )

        result = module.minimize_alm(
            np.array([0.0]),
            ["x_upper_bound"],
            evaluate_problem,
            settings,
            {"maxiter": 50, "maxcor": 20, "ftol": 1e-12, "gtol": 1e-12},
        )

        self.assertFalse(result.success)
        self.assertLessEqual(result.x[0], 1.0 + 1e-6)
        self.assertLessEqual(result.constraint_values[0], 1e-6)
        self.assertAlmostEqual(result.final_raw_stationarity_norm, 1.0)
        self.assertAlmostEqual(result.final_kkt_stationarity_norm, 0.0)

    def test_minimize_alm_failure_reports_last_solved_subproblem_state(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-12,
            stationarity_tol=1e-12,
        )

        def evaluate_problem(x, multipliers, penalty):
            value = 0.5 * (x[0] - 2.0) ** 2
            grad = np.array([x[0] - 2.0])
            constraint_value = module.upper_bound_residual(x[0], 1.0)
            constraint_grad = (
                np.array([1.0]) if constraint_value > 0.0 else np.array([0.0])
            )
            return module.augmented_inequality_objective(
                value,
                grad,
                [constraint_value],
                [constraint_grad],
                multipliers,
                penalty,
            )

        result = module.minimize_alm(
            np.array([0.0]),
            ["x_upper_bound"],
            evaluate_problem,
            settings,
            {"maxiter": 50, "maxcor": 20, "ftol": 1e-12, "gtol": 1e-12},
        )

        self.assertFalse(result.success)
        self.assertAlmostEqual(result.penalty, 1.0)
        self.assertEqual(result.multipliers, [0.0])


class InitOnlyResultTests(unittest.TestCase):
    def test_published_surface_mode_runs_search_topology_gate_at_strict_scale(self):
        module = load_single_stage_example_module()
        contract = module.resolve_surface_mode_contract(
            SimpleNamespace(
                surface_mode=module.PUBLISHED_MULTISURFACE,
                num_surfaces=1,
                inner_surface_ratio=0.8,
            ),
            warn_on_legacy_mapping=False,
        )

        with (
            patch.object(module, "TOPOLOGY_GATE_FIELDLINES", 4),
            patch.object(
                module,
                "safe_evaluate_topology_gate",
                return_value={"enabled": True, "success": True, "gate_scale": 1.0},
            ) as gate,
        ):
            status = module.evaluate_search_topology_gate(
                3,
                object(),
                object(),
                surface_mode_contract=contract,
            )

        gate.assert_called_once()
        self.assertTrue(status["enabled"])
        self.assertEqual(status["gate_scale"], 1.0)

    def test_final_topology_gate_for_results_skips_expensive_probe_in_init_only(self):
        module = load_single_stage_example_module()

        with patch.object(
            module,
            "evaluate_search_topology_gate",
            side_effect=AssertionError("should not run"),
        ):
            status = module.final_topology_gate_for_results(True, 2, object(), object())

        self.assertFalse(status["evaluated"])
        self.assertIsNone(status["success"])
        self.assertIsNone(status["stop_reason_counts"])


if __name__ == "__main__":
    unittest.main()
