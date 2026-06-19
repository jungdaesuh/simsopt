import importlib.util
import sys
import unittest
import uuid
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
SURFACE_MODE_CONTRACTS_PATH = EXAMPLE_ROOT / "banana_opt" / "surface_mode_contracts.py"
SINGLE_STAGE_ENTRYPOINT_PATH = (
    EXAMPLE_ROOT / "SINGLE_STAGE" / "single_stage_banana_example.py"
)
GOAL_MODE_COMPARISON_PATH = EXAMPLE_ROOT / "run_single_stage_goal_mode_comparison.py"


def load_module(path: Path, stem: str, *, register_in_sys_modules: bool = False):
    spec = importlib.util.spec_from_file_location(f"{stem}_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    if register_in_sys_modules:
        sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_surface_mode_contracts_module():
    return load_module(
        SURFACE_MODE_CONTRACTS_PATH,
        "surface_mode_contracts",
        register_in_sys_modules=True,
    )


def load_single_stage_example_module():
    return load_module(
        SINGLE_STAGE_ENTRYPOINT_PATH,
        "single_stage_banana_example",
        register_in_sys_modules=True,
    )


def load_goal_mode_comparison_module():
    return load_module(
        GOAL_MODE_COMPARISON_PATH,
        "run_single_stage_goal_mode_comparison",
    )


def make_surface_mode_args(**overrides):
    values = {
        "surface_mode": None,
        "num_surfaces": 1,
        "inner_surface_ratio": 0.72,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def resolve_explicit_multisurface_contract(module, **arg_overrides):
    args = make_surface_mode_args(
        surface_mode=module.EXPERIMENTAL_MULTISURFACE,
        **arg_overrides,
    )
    contract = module.resolve_surface_mode_contract(args, warn_on_legacy_mapping=False)
    return args, contract


class SurfaceModeContractTests(unittest.TestCase):
    def test_legacy_two_surface_mapping_warns_and_preserves_legacy_fields(self):
        module = load_surface_mode_contracts_module()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            contract = module.build_surface_mode_contract(
                requested_surface_mode=None,
                legacy_num_surfaces=2,
                legacy_inner_surface_ratio=0.7,
            )

        self.assertEqual(contract.mode, module.EXPERIMENTAL_MULTISURFACE)
        self.assertEqual(
            contract.source, module.SURFACE_MODE_SOURCE_LEGACY_NUM_SURFACES_MAPPING
        )
        self.assertEqual(contract.label_fractions, (0.7, 1.0))
        self.assertEqual(contract.weights, (1.0, 1.0))
        self.assertEqual(contract.legacy_num_surfaces, 2)
        self.assertEqual(contract.legacy_inner_surface_ratio, 0.7)
        self.assertTrue(
            any(issubclass(entry.category, DeprecationWarning) for entry in caught)
        )

    def test_explicit_surface_mode_uses_effective_contract_and_clears_legacy_fields(
        self,
    ):
        module = load_surface_mode_contracts_module()

        contract = module.build_surface_mode_contract(
            requested_surface_mode=module.EXPERIMENTAL_MULTISURFACE,
            legacy_num_surfaces=1,
            legacy_inner_surface_ratio=0.65,
        )
        metadata = module.build_surface_mode_metadata(contract)

        self.assertEqual(contract.mode, module.EXPERIMENTAL_MULTISURFACE)
        self.assertEqual(contract.source, module.SURFACE_MODE_SOURCE_EXPLICIT_CLI)
        self.assertEqual(contract.label_fractions, (0.65, 1.0))
        self.assertIsNone(contract.legacy_num_surfaces)
        self.assertIsNone(contract.legacy_inner_surface_ratio)
        self.assertEqual(metadata["SURFACE_MODE"], module.EXPERIMENTAL_MULTISURFACE)
        self.assertEqual(metadata["SURFACE_LABEL_FRACTIONS"], [0.65, 1.0])
        self.assertEqual(metadata["SURFACE_WEIGHTS"], [1.0, 1.0])
        self.assertIsNone(metadata["LEGACY_NUM_SURFACES"])
        self.assertIsNone(metadata["LEGACY_INNER_SURFACE_RATIO"])

    def test_published_multisurface_contract_has_fixed_stack_and_runtime_support(self):
        module = load_surface_mode_contracts_module()

        contract = module.build_surface_mode_contract(
            requested_surface_mode=module.PUBLISHED_MULTISURFACE,
            legacy_num_surfaces=1,
            legacy_inner_surface_ratio=0.8,
        )

        self.assertEqual(contract.label_fractions, (0.6, 0.8, 1.0))
        self.assertEqual(contract.weights, (1.0, 1.0, 1.0))
        self.assertEqual(
            contract.stack_policy, module.SURFACE_STACK_POLICY_PUBLISHED_FIXED_STACK
        )
        self.assertEqual(
            module.surface_mode_surface_names(contract),
            ("inner0", "inner1", "outer"),
        )
        self.assertFalse(contract.requires_inner_surface_ratio)
        self.assertEqual(
            contract.surface_count_policy,
            module.SURFACE_COUNT_POLICY_PUBLISHED_FIXED_STACK_V1,
        )
        self.assertEqual(
            contract.final_refinement_policy,
            module.FINAL_REFINEMENT_POLICY_UNSUPPORTED,
        )
        self.assertEqual(contract.current_policy, module.CURRENT_POLICY_VACUUM_LOCKED)
        self.assertEqual(contract.topology_policy, module.TOPOLOGY_POLICY_SEARCH_GATE)
        self.assertEqual(
            contract.production_support_level,
            module.PRODUCTION_SUPPORT_LEVEL_PUBLISHED_V1,
        )
        module.validate_surface_mode_runtime_support(contract)

    def test_experimental_multisurface_contract_has_fail_closed_runtime_policy(self):
        module = load_surface_mode_contracts_module()

        contract = module.build_surface_mode_contract(
            requested_surface_mode=module.EXPERIMENTAL_MULTISURFACE,
            legacy_num_surfaces=1,
            legacy_inner_surface_ratio=0.72,
        )
        metadata = module.build_surface_mode_metadata(contract)

        self.assertTrue(contract.requires_inner_surface_ratio)
        self.assertEqual(
            contract.surface_count_policy,
            module.SURFACE_COUNT_POLICY_EXPERIMENTAL_TWO_SURFACES,
        )
        self.assertEqual(
            contract.final_refinement_policy,
            module.FINAL_REFINEMENT_POLICY_UNSUPPORTED,
        )
        self.assertEqual(contract.current_policy, module.CURRENT_POLICY_INHERIT_STAGE2)
        self.assertEqual(contract.topology_policy, module.TOPOLOGY_POLICY_SEARCH_GATE)
        self.assertEqual(
            contract.production_support_level,
            module.PRODUCTION_SUPPORT_LEVEL_EXPERIMENTAL,
        )
        self.assertEqual(
            metadata["SURFACE_MODE_TELEMETRY_SCHEMA_VERSION"],
            module.SURFACE_MODE_TELEMETRY_SCHEMA_VERSION_V1,
        )
        module.validate_surface_mode_runtime_support(contract)

    def test_runtime_support_rejects_contract_policy_drift(self):
        module = load_surface_mode_contracts_module()
        contract = module.build_surface_mode_contract(
            requested_surface_mode=module.EXPERIMENTAL_MULTISURFACE,
            legacy_num_surfaces=1,
            legacy_inner_surface_ratio=0.72,
        )
        broken = replace(
            contract,
            surface_count_policy=module.SURFACE_COUNT_POLICY_SINGLE_SURFACE,
        )

        self.assertFalse(module.surface_mode_runtime_supported(broken))
        with self.assertRaisesRegex(ValueError, "surface_count_policy"):
            module.validate_surface_mode_runtime_support(broken)

    def test_runtime_support_rejects_experimental_surface_shape_drift(self):
        module = load_surface_mode_contracts_module()
        contract = module.build_surface_mode_contract(
            requested_surface_mode=module.EXPERIMENTAL_MULTISURFACE,
            legacy_num_surfaces=1,
            legacy_inner_surface_ratio=0.72,
        )
        broken = replace(contract, label_fractions=(1.0,), weights=(1.0,))

        self.assertFalse(module.surface_mode_runtime_supported(broken))
        with self.assertRaisesRegex(ValueError, "num_surfaces"):
            module.validate_surface_mode_runtime_support(broken)

    def test_contract_capability_helpers_fail_closed_on_policy_drift(self):
        module = load_surface_mode_contracts_module()
        single_surface_contract = module.build_surface_mode_contract(
            requested_surface_mode=module.SINGLE_SURFACE,
            legacy_num_surfaces=1,
            legacy_inner_surface_ratio=0.72,
        )
        no_refinement_contract = replace(
            single_surface_contract,
            final_refinement_policy=module.FINAL_REFINEMENT_POLICY_UNSUPPORTED,
        )
        experimental_contract = module.build_surface_mode_contract(
            requested_surface_mode=module.EXPERIMENTAL_MULTISURFACE,
            legacy_num_surfaces=1,
            legacy_inner_surface_ratio=0.72,
        )
        no_topology_contract = replace(
            experimental_contract,
            topology_policy=module.TOPOLOGY_POLICY_UNSUPPORTED,
        )

        self.assertFalse(
            module.surface_mode_supports_boozer_stage_refinement(no_refinement_contract)
        )
        self.assertFalse(
            module.surface_mode_supports_topology_gate(no_topology_contract)
        )

    def test_alm_supports_all_surface_modes(self):
        module = load_surface_mode_contracts_module()

        self.assertTrue(module.surface_mode_supports_alm(module.SINGLE_SURFACE))
        self.assertTrue(
            module.surface_mode_supports_alm(module.EXPERIMENTAL_MULTISURFACE)
        )
        self.assertTrue(module.surface_mode_supports_alm(module.PUBLISHED_MULTISURFACE))

    def test_topology_gate_supports_published_and_experimental_modes(self):
        module = load_surface_mode_contracts_module()

        self.assertFalse(
            module.surface_mode_supports_topology_gate(module.SINGLE_SURFACE)
        )
        self.assertTrue(
            module.surface_mode_supports_topology_gate(module.PUBLISHED_MULTISURFACE)
        )
        self.assertTrue(
            module.surface_mode_supports_topology_gate(module.EXPERIMENTAL_MULTISURFACE)
        )


class SingleStageSurfaceModeIntegrationTests(unittest.TestCase):
    def test_resolve_surface_mode_contract_suppresses_legacy_warning_for_default_single_surface(
        self,
    ):
        module = load_single_stage_example_module()
        contracts_module = load_surface_mode_contracts_module()
        args = make_surface_mode_args()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            contract = module.resolve_surface_mode_contract(args)

        self.assertEqual(contract.mode, module.SINGLE_SURFACE)
        self.assertEqual(
            contract.source,
            contracts_module.SURFACE_MODE_SOURCE_LEGACY_NUM_SURFACES_MAPPING,
        )
        self.assertEqual(len(caught), 0)

    def test_resolve_surface_mode_contract_prefers_explicit_surface_mode(self):
        module = load_single_stage_example_module()
        args = make_surface_mode_args(surface_mode=module.EXPERIMENTAL_MULTISURFACE)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            contract = module.resolve_surface_mode_contract(args)

        self.assertEqual(contract.mode, module.EXPERIMENTAL_MULTISURFACE)
        self.assertEqual(contract.num_surfaces, 2)
        self.assertEqual(contract.label_fractions, (0.72, 1.0))
        self.assertEqual(len(caught), 0)

    def test_make_run_identity_config_uses_effective_surface_contract(self):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            boozer_stage_refinement=False,
            refinement_boozer_stage="final",
            refinement_maxiter=20,
            refinement_chunk_maxiter=10,
            refinement_max_stalled_chunks=2,
            alm_formulation="weighted_sum",
            alm_qs_threshold=None,
            alm_boozer_threshold=None,
            alm_iota_penalty_threshold=None,
            alm_length_penalty_threshold=None,
            single_stage_goal_mode="target",
            cc_dist=0.05,
            cc_weight=100.0,
            single_stage_poloidal_weight=1.0,
            single_stage_width_weight=1.0,
            single_stage_selfint_weight=1.0,
            single_stage_hardware_keepout_weight=0.0,
            hardware_keepout_json=None,
            single_stage_vessel_keepout_weight=0.0,
            single_stage_vessel_keepout_clearance=0.0,
            single_stage_available_envelope_reward_weight=0.0,
            single_stage_hardware_sdf_free_space_reward_weight=0.0,
            clearance_hinge_weight=0.0,
            clearance_hinge_margin_m=0.005,
            single_stage_poloidal_threshold_rad=1.0,
            single_stage_width_min_threshold=0.01,
            single_stage_width_max_threshold=0.06,
            curvature_weight=0.1,
            curvature_threshold=100.0,
            banana_current_max_A=1.6e4,
            init_only=False,
            basin_hops=0,
            basin_stepsize=0.01,
            basin_temperature=1.0,
            basin_niter_success=0,
            ftol=1.0e-15,
            gtol=1.0e-15,
            alm_max_outer_iters=10,
            alm_penalty_init=1.0,
            alm_penalty_scale=10.0,
            alm_penalty_max=1.0e8,
            alm_feas_tol=1.0e-6,
            alm_stationarity_tol=1.0e-6,
            num_surfaces=1,
            inner_surface_ratio=0.73,
            surface_gap_threshold=0.0,
            multisurface_ramp_iterations=5,
            inner_surface_initial_weight=0.0,
            multisurface_initial_step_scale=1.0,
            multisurface_initial_step_maxiter=0,
            topology_gate_fieldlines=4,
            topology_gate_tmax=2.0,
            topology_gate_tol=1.0e-7,
            topology_gate_survival_threshold=0.25,
            topology_gate_penalty_scale=4.0,
            hardware_search_mode="hard",
            hardware_search_soft_iterations=0,
            curvature_traversal_band=0.0,
            curvature_traversal_eval_budget=0,
            topology_scorer_every=0,
            topology_scorer_nfieldlines=12,
            topology_scorer_tmax=50.0,
            topology_scorer_min_returns=256,
            confinement_objective_weight=0.0,
            confinement_surrogate_worst_k=3,
            confinement_surrogate_early_threshold=0.2,
            confinement_surrogate_mean_weight=0.2,
            confinement_surrogate_worst_weight=0.6,
            confinement_surrogate_early_weight=0.2,
            alm_trust_radius_init=0.05,
            alm_trust_radius_min=1.0e-4,
            alm_trust_radius_shrink=0.5,
            alm_trust_radius_grow=1.5,
            alm_max_inner_attempts=4,
            alm_max_subproblem_continuations=20,
            alm_distance_smoothing=0.005,
            alm_curvature_smoothing=0.05,
            seed_regime="auto",
            surface_mode="experimental_multisurface",
        )
        contract = module.resolve_surface_mode_contract(
            args, warn_on_legacy_mapping=False
        )

        config = module.make_run_identity_config(
            args,
            stage2_bs_path="seed/biot_savart_opt.json",
            stage="initial",
            constraint_weight=1.0,
            constraint_method="penalty",
            vol_target=0.1,
            iota_target=0.15,
            boozer_I=0.0,
            plasma_current_A=0.0,
            banana_surf_radius=0.2,
            nphi=255,
            ntheta=64,
            rng_seed=7,
            surface_mode_contract=contract,
            effective_num_surfaces=contract.num_surfaces,
            effective_inner_surface_ratio=module.resolve_surface_mode_inner_surface_ratio(
                contract,
                fallback_inner_surface_ratio=args.inner_surface_ratio,
            ),
        )

        self.assertEqual(config.num_surfaces, 2)
        self.assertEqual(config.surface_mode, module.EXPERIMENTAL_MULTISURFACE)
        self.assertEqual(config.surface_label_fractions, (0.73, 1.0))
        self.assertEqual(config.inner_surface_ratio, 0.73)

        published_args = SimpleNamespace(**vars(args))
        published_args.surface_mode = module.PUBLISHED_MULTISURFACE
        published_contract = module.resolve_surface_mode_contract(
            published_args,
            warn_on_legacy_mapping=False,
        )
        published_config = module.make_run_identity_config(
            published_args,
            stage2_bs_path="seed/biot_savart_opt.json",
            stage="initial",
            constraint_weight=1.0,
            constraint_method="penalty",
            vol_target=0.1,
            iota_target=0.15,
            boozer_I=0.0,
            plasma_current_A=0.0,
            banana_surf_radius=0.2,
            nphi=255,
            ntheta=64,
            rng_seed=7,
            surface_mode_contract=published_contract,
            effective_num_surfaces=published_contract.num_surfaces,
            effective_inner_surface_ratio=module.resolve_surface_mode_inner_surface_ratio(
                published_contract,
                fallback_inner_surface_ratio=published_args.inner_surface_ratio,
            ),
        )

        self.assertEqual(published_config.num_surfaces, 3)
        self.assertEqual(published_config.surface_mode, module.PUBLISHED_MULTISURFACE)
        self.assertEqual(
            published_config.surface_label_fractions,
            (0.6, 0.8, 1.0),
        )
        self.assertNotEqual(
            config.surface_label_fractions,
            published_config.surface_label_fractions,
        )

    def test_validate_boozer_stage_refinement_args_rejects_explicit_multisurface_contract(
        self,
    ):
        module = load_single_stage_example_module()
        args, contract = resolve_explicit_multisurface_contract(
            module,
            boozer_stage_refinement=True,
            constraint_method="penalty",
            basin_hops=0,
            boozer_stage="initial",
            refinement_boozer_stage="final",
            refinement_maxiter=20,
            refinement_chunk_maxiter=10,
            refinement_max_stalled_chunks=2,
        )

        with self.assertRaisesRegex(
            ValueError,
            f"--surface-mode={module.SINGLE_SURFACE}",
        ):
            module.validate_boozer_stage_refinement_args(
                args,
                constraint_weight=1.0,
                surface_mode_contract=contract,
            )

    def test_validate_surface_mode_constraint_args_allows_experimental_multisurface_alm(
        self,
    ):
        module = load_single_stage_example_module()
        args, contract = resolve_explicit_multisurface_contract(
            module,
            constraint_method="alm",
        )

        module.validate_surface_mode_constraint_args(
            args,
            surface_mode_contract=contract,
        )

    def test_validate_surface_mode_constraint_args_allows_published_multisurface_alm(
        self,
    ):
        module = load_single_stage_example_module()
        args = make_surface_mode_args(
            surface_mode=module.PUBLISHED_MULTISURFACE,
            constraint_method="alm",
        )
        contract = module.resolve_surface_mode_contract(
            args, warn_on_legacy_mapping=False
        )

        module.validate_surface_mode_constraint_args(
            args,
            surface_mode_contract=contract,
        )

    def test_validate_surface_mode_constraint_args_allows_published_frontier(self):
        module = load_single_stage_example_module()
        args = make_surface_mode_args(
            surface_mode=module.PUBLISHED_MULTISURFACE,
            constraint_method="penalty",
            single_stage_goal_mode="frontier",
            magnetic_well_weight=1.0,
        )
        contract = module.resolve_surface_mode_contract(
            args,
            warn_on_legacy_mapping=False,
        )

        module.validate_surface_mode_constraint_args(
            args,
            surface_mode_contract=contract,
        )

    def test_validate_surface_mode_constraint_args_rejects_single_surface_magnetic_well(
        self,
    ):
        module = load_single_stage_example_module()
        args = make_surface_mode_args(
            surface_mode=module.SINGLE_SURFACE,
            constraint_method="penalty",
            magnetic_well_weight=1.0,
        )
        contract = module.resolve_surface_mode_contract(
            args,
            warn_on_legacy_mapping=False,
        )

        with self.assertRaisesRegex(ValueError, "at least three Boozer surfaces"):
            module.validate_surface_mode_constraint_args(
                args,
                surface_mode_contract=contract,
            )

    def test_resolve_plasma_current_settings_rejects_published_nonzero_default_current(
        self,
    ):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            boozer_I=None,
            plasma_current_A=None,
            finite_current_mode=module.DEFAULT_FINITE_CURRENT_MODE,
        )
        contract = module.resolve_surface_mode_contract(
            make_surface_mode_args(surface_mode=module.PUBLISHED_MULTISURFACE),
            warn_on_legacy_mapping=False,
        )

        with self.assertRaisesRegex(ValueError, "vacuum-locked"):
            module.resolve_plasma_current_settings(
                args,
                finite_current_mode=module.DEFAULT_FINITE_CURRENT_MODE,
                default_plasma_current_A=1.0,
                surface_mode_contract=contract,
            )

    def test_resolve_plasma_current_settings_normalizes_published_default_mode_to_vacuum(
        self,
    ):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            boozer_I=None,
            plasma_current_A=None,
            finite_current_mode=module.DEFAULT_FINITE_CURRENT_MODE,
        )
        contract = module.resolve_surface_mode_contract(
            make_surface_mode_args(surface_mode=module.PUBLISHED_MULTISURFACE),
            warn_on_legacy_mapping=False,
        )

        settings = module.resolve_plasma_current_settings(
            args,
            finite_current_mode=module.DEFAULT_FINITE_CURRENT_MODE,
            default_plasma_current_A=0.0,
            surface_mode_contract=contract,
        )

        self.assertEqual(settings["mode"], "vacuum")
        self.assertEqual(settings["effective_mode"], "vacuum")
        self.assertEqual(settings["plasma_current_A"], 0.0)
        self.assertEqual(settings["boozer_I"], 0.0)

    def test_resolve_plasma_current_settings_rejects_published_requested_finite_mode(
        self,
    ):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            boozer_I=None,
            plasma_current_A=None,
            finite_current_mode="jhalpern30_proxy_field",
        )
        contract = module.resolve_surface_mode_contract(
            make_surface_mode_args(surface_mode=module.PUBLISHED_MULTISURFACE),
            warn_on_legacy_mapping=False,
        )

        with self.assertRaisesRegex(ValueError, "vacuum-locked"):
            module.resolve_plasma_current_settings(
                args,
                finite_current_mode=module.DEFAULT_FINITE_CURRENT_MODE,
                default_plasma_current_A=0.0,
                surface_mode_contract=contract,
            )

    def test_resolve_plasma_current_settings_rejects_published_inherited_finite_mode(
        self,
    ):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            boozer_I=None,
            plasma_current_A=None,
            finite_current_mode=None,
        )
        contract = module.resolve_surface_mode_contract(
            make_surface_mode_args(surface_mode=module.PUBLISHED_MULTISURFACE),
            warn_on_legacy_mapping=False,
        )

        with self.assertRaisesRegex(ValueError, "finite-current donor mode"):
            module.resolve_plasma_current_settings(
                args,
                finite_current_mode="jhalpern30_proxy_field",
                default_plasma_current_A=0.0,
                surface_mode_contract=contract,
            )

    def test_resolve_plasma_current_settings_rejects_published_raw_boozer_I(self):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            boozer_I=0.1,
            plasma_current_A=None,
            finite_current_mode=module.DEFAULT_FINITE_CURRENT_MODE,
        )
        contract = module.resolve_surface_mode_contract(
            make_surface_mode_args(surface_mode=module.PUBLISHED_MULTISURFACE),
            warn_on_legacy_mapping=False,
        )

        with self.assertRaisesRegex(ValueError, "boozer-I"):
            module.resolve_plasma_current_settings(
                args,
                finite_current_mode=module.DEFAULT_FINITE_CURRENT_MODE,
                default_plasma_current_A=0.0,
                surface_mode_contract=contract,
            )

    def test_resolve_plasma_current_settings_rejects_published_nonzero_cli_current(
        self,
    ):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            boozer_I=None,
            plasma_current_A=1.0,
            finite_current_mode=module.DEFAULT_FINITE_CURRENT_MODE,
        )
        contract = module.resolve_surface_mode_contract(
            make_surface_mode_args(surface_mode=module.PUBLISHED_MULTISURFACE),
            warn_on_legacy_mapping=False,
        )

        with self.assertRaisesRegex(ValueError, "plasma-current-A"):
            module.resolve_plasma_current_settings(
                args,
                finite_current_mode=module.DEFAULT_FINITE_CURRENT_MODE,
                default_plasma_current_A=0.0,
                surface_mode_contract=contract,
            )


class GoalModeWrapperSurfaceModeTests(unittest.TestCase):
    def test_goal_mode_parser_reuses_surface_mode_ssot_choices(self):
        module = load_goal_mode_comparison_module()
        contracts_module = load_surface_mode_contracts_module()
        parser = module.build_parser()

        self.assertEqual(
            tuple(parser._option_string_actions["--surface-mode"].choices),
            contracts_module.SURFACE_MODE_CHOICES,
        )

    def test_goal_mode_command_forwards_surface_mode_flag(self):
        module = load_goal_mode_comparison_module()
        contracts_module = load_surface_mode_contracts_module()
        args = module.build_parser().parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
                "--surface-mode",
                contracts_module.EXPERIMENTAL_MULTISURFACE,
                "--num-surfaces",
                "1",
                "--inner-surface-ratio",
                "0.75",
            ]
        )

        command = module.build_single_stage_goal_mode_command(
            args,
            goal_mode="target",
            stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
            case_output_root=Path("/tmp/out"),
        )

        self.assertIn("--surface-mode", command)
        self.assertIn(contracts_module.EXPERIMENTAL_MULTISURFACE, command)


class _MonotoneVolumeSurface:
    """Minimal SurfaceRZFourier stand-in for the config builder.

    ``volume()`` is a strictly increasing function of the normalized-flux label
    ``s`` and scales with the major-radius rescaling, so the real
    ``build_surface_configs_for_contract`` volume-ordered target derivation and
    strict inner-to-outer ordering check are exercised faithfully without a
    full VMEC equilibrium.
    """

    def __init__(self, s):
        self._s = float(s)
        self._scale = 1.0

    def major_radius(self):
        return 1.0

    def get_dofs(self):
        return np.array([1.0])

    def set_dofs(self, dofs):
        self._scale = float(np.asarray(dofs)[0])

    def volume(self):
        return (self._s**1.5) * (self._scale**3) * 10.0


class _MonotoneVolumeSurfaceFactory:
    @staticmethod
    def from_wout(file_loc, range, nphi, ntheta, s):  # noqa: A002 - simsopt API
        return _MonotoneVolumeSurface(s)


def load_single_stage_geometry_module():
    if str(EXAMPLE_ROOT) not in sys.path:
        sys.path.insert(0, str(EXAMPLE_ROOT))
    return load_module(
        EXAMPLE_ROOT / "banana_opt" / "single_stage_geometry.py",
        "single_stage_geometry",
        register_in_sys_modules=True,
    )


class PublishedSurfaceDepthConfigurationTests(unittest.TestCase):
    """Item #13: opt-in interior-covering published nested-surface stack."""

    def test_default_published_stack_is_byte_identical_lock(self):
        module = load_surface_mode_contracts_module()

        # No depth knob set: the resolved published contract must match the
        # current v1 3-shell stack byte-for-byte (count, fractions, names,
        # weights, vacuum-lock current policy, and physics-contract surface
        # count). This is the default-off regression lock.
        default_contract = module.build_surface_mode_contract(
            requested_surface_mode=module.PUBLISHED_MULTISURFACE,
            legacy_num_surfaces=1,
            legacy_inner_surface_ratio=0.8,
        )
        explicit_default_preset = module.build_surface_mode_contract(
            requested_surface_mode=module.PUBLISHED_MULTISURFACE,
            legacy_num_surfaces=1,
            legacy_inner_surface_ratio=0.8,
            published_surface_preset=module.PUBLISHED_PRESET_DEFAULT_V1,
        )

        for contract in (default_contract, explicit_default_preset):
            self.assertEqual(contract.label_fractions, (0.6, 0.8, 1.0))
            self.assertEqual(contract.weights, (1.0, 1.0, 1.0))
            self.assertEqual(contract.num_surfaces, 3)
            self.assertEqual(
                module.surface_mode_surface_names(contract),
                ("inner0", "inner1", "outer"),
            )
            self.assertEqual(
                contract.current_policy, module.CURRENT_POLICY_VACUUM_LOCKED
            )
            module.validate_surface_mode_runtime_support(contract)

        # The explicit default-v1 preset and the no-knob default resolve to an
        # identical contract.
        self.assertEqual(default_contract, explicit_default_preset)

    def test_interior_covering_preset_adds_inner_shell_toward_core(self):
        module = load_surface_mode_contracts_module()

        contract = module.build_surface_mode_contract(
            requested_surface_mode=module.PUBLISHED_MULTISURFACE,
            legacy_num_surfaces=1,
            legacy_inner_surface_ratio=0.8,
            published_surface_preset=module.PUBLISHED_PRESET_INTERIOR_COVERING,
        )

        self.assertEqual(contract.label_fractions, (0.4, 0.6, 0.8, 1.0))
        self.assertEqual(contract.weights, (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(contract.num_surfaces, 4)
        self.assertEqual(
            module.surface_mode_surface_names(contract),
            ("inner0", "inner1", "inner2", "outer"),
        )
        # Strictly increasing in (0, 1] with the outermost exactly 1.0, and the
        # innermost shell reaches below the v1 floor of 0.6 toward the core.
        fractions = contract.label_fractions
        self.assertTrue(all(0.0 < value <= 1.0 for value in fractions))
        self.assertTrue(
            all(left < right for left, right in zip(fractions[:-1], fractions[1:]))
        )
        self.assertEqual(fractions[-1], 1.0)
        self.assertLess(fractions[0], 0.6)
        module.validate_surface_mode_runtime_support(contract)

    def test_deep_preset_reaches_fraction_0p2(self):
        module = load_surface_mode_contracts_module()

        contract = module.build_surface_mode_contract(
            requested_surface_mode=module.PUBLISHED_MULTISURFACE,
            legacy_num_surfaces=1,
            legacy_inner_surface_ratio=0.8,
            published_surface_preset=module.PUBLISHED_PRESET_INTERIOR_COVERING_DEEP,
        )

        self.assertEqual(contract.label_fractions, (0.2, 0.4, 0.6, 0.8, 1.0))
        self.assertEqual(contract.num_surfaces, 5)
        module.validate_surface_mode_runtime_support(contract)

    def test_explicit_fractions_produce_valid_interior_stack(self):
        module = load_surface_mode_contracts_module()

        contract = module.build_surface_mode_contract(
            requested_surface_mode=module.PUBLISHED_MULTISURFACE,
            legacy_num_surfaces=1,
            legacy_inner_surface_ratio=0.8,
            published_label_fractions=(0.25, 0.5, 0.75, 1.0),
        )

        self.assertEqual(contract.label_fractions, (0.25, 0.5, 0.75, 1.0))
        self.assertEqual(contract.weights, (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(
            module.surface_mode_surface_names(contract),
            ("inner0", "inner1", "inner2", "outer"),
        )
        module.validate_surface_mode_runtime_support(contract)

    def test_explicit_fractions_take_precedence_over_default_preset(self):
        module = load_surface_mode_contracts_module()

        contract = module.build_surface_mode_contract(
            requested_surface_mode=module.PUBLISHED_MULTISURFACE,
            legacy_num_surfaces=1,
            legacy_inner_surface_ratio=0.8,
            published_surface_preset=module.PUBLISHED_PRESET_DEFAULT_V1,
            published_label_fractions=(0.3, 0.65, 1.0),
        )

        self.assertEqual(contract.label_fractions, (0.3, 0.65, 1.0))

    def test_non_monotone_fractions_rejected(self):
        module = load_surface_mode_contracts_module()

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            module.build_surface_mode_contract(
                requested_surface_mode=module.PUBLISHED_MULTISURFACE,
                legacy_num_surfaces=1,
                legacy_inner_surface_ratio=0.8,
                published_label_fractions=(0.8, 0.6, 1.0),
            )

    def test_outermost_not_one_rejected(self):
        module = load_surface_mode_contracts_module()

        with self.assertRaisesRegex(ValueError, "exactly 1.0"):
            module.build_surface_mode_contract(
                requested_surface_mode=module.PUBLISHED_MULTISURFACE,
                legacy_num_surfaces=1,
                legacy_inner_surface_ratio=0.8,
                published_label_fractions=(0.6, 0.8, 0.95),
            )

    def test_empty_fractions_rejected(self):
        module = load_surface_mode_contracts_module()

        with self.assertRaisesRegex(ValueError, "at least one surface"):
            module.build_surface_mode_contract(
                requested_surface_mode=module.PUBLISHED_MULTISURFACE,
                legacy_num_surfaces=1,
                legacy_inner_surface_ratio=0.8,
                published_label_fractions=(),
            )

    def test_nonpositive_fraction_rejected(self):
        module = load_surface_mode_contracts_module()

        with self.assertRaisesRegex(ValueError, r"\(0, 1\]"):
            module.build_surface_mode_contract(
                requested_surface_mode=module.PUBLISHED_MULTISURFACE,
                legacy_num_surfaces=1,
                legacy_inner_surface_ratio=0.8,
                published_label_fractions=(0.0, 0.5, 1.0),
            )

    def test_unknown_preset_rejected(self):
        module = load_surface_mode_contracts_module()

        with self.assertRaisesRegex(ValueError, "Unsupported published surface preset"):
            module.build_surface_mode_contract(
                requested_surface_mode=module.PUBLISHED_MULTISURFACE,
                legacy_num_surfaces=1,
                legacy_inner_surface_ratio=0.8,
                published_surface_preset="does_not_exist",
            )

    def test_explicit_fractions_and_nondefault_preset_together_rejected(self):
        module = load_surface_mode_contracts_module()

        with self.assertRaisesRegex(ValueError, "not both"):
            module.build_surface_mode_contract(
                requested_surface_mode=module.PUBLISHED_MULTISURFACE,
                legacy_num_surfaces=1,
                legacy_inner_surface_ratio=0.8,
                published_surface_preset=module.PUBLISHED_PRESET_INTERIOR_COVERING,
                published_label_fractions=(0.6, 0.8, 1.0),
            )

    def test_depth_knob_on_non_published_mode_rejected(self):
        module = load_surface_mode_contracts_module()

        with self.assertRaisesRegex(ValueError, "only apply to"):
            module.build_surface_mode_contract(
                requested_surface_mode=module.SINGLE_SURFACE,
                legacy_num_surfaces=1,
                legacy_inner_surface_ratio=0.8,
                published_surface_preset=module.PUBLISHED_PRESET_INTERIOR_COVERING,
            )

    def test_runtime_support_rejects_mutated_nonmonotone_published_stack(self):
        module = load_surface_mode_contracts_module()
        contract = module.build_surface_mode_contract(
            requested_surface_mode=module.PUBLISHED_MULTISURFACE,
            legacy_num_surfaces=1,
            legacy_inner_surface_ratio=0.8,
            published_surface_preset=module.PUBLISHED_PRESET_INTERIOR_COVERING,
        )
        broken = replace(contract, label_fractions=(0.6, 0.4, 0.8, 1.0))

        self.assertFalse(module.surface_mode_runtime_supported(broken))
        with self.assertRaisesRegex(ValueError, "label_fractions invalid"):
            module.validate_surface_mode_runtime_support(broken)

    def test_runtime_support_rejects_mutated_outer_fraction(self):
        module = load_surface_mode_contracts_module()
        contract = module.build_surface_mode_contract(
            requested_surface_mode=module.PUBLISHED_MULTISURFACE,
            legacy_num_surfaces=1,
            legacy_inner_surface_ratio=0.8,
            published_surface_preset=module.PUBLISHED_PRESET_INTERIOR_COVERING,
        )
        broken = replace(contract, label_fractions=(0.4, 0.6, 0.8, 0.95))

        self.assertFalse(module.surface_mode_runtime_supported(broken))

    def test_published_surface_names_generator_matches_v1_default(self):
        module = load_surface_mode_contracts_module()

        self.assertEqual(
            module.published_surface_names(3), ("inner0", "inner1", "outer")
        )
        self.assertEqual(module.published_surface_names(1), ("outer",))
        self.assertEqual(
            module.published_surface_names(5),
            ("inner0", "inner1", "inner2", "inner3", "outer"),
        )
        with self.assertRaisesRegex(ValueError, "at least one surface"):
            module.published_surface_names(0)

    def test_build_surface_configs_for_contract_builds_volume_ordered_interior_stack(
        self,
    ):
        contracts_module = load_surface_mode_contracts_module()
        geometry_module = load_single_stage_geometry_module()

        seed_label = 0.9
        outer_target_volume = 0.5
        contract = contracts_module.build_surface_mode_contract(
            requested_surface_mode=contracts_module.PUBLISHED_MULTISURFACE,
            legacy_num_surfaces=1,
            legacy_inner_surface_ratio=0.8,
            published_surface_preset=(
                contracts_module.PUBLISHED_PRESET_INTERIOR_COVERING
            ),
        )

        configs = geometry_module.build_surface_configs_for_contract(
            "dummy.nc",
            8,
            8,
            seed_label,
            1.0,
            outer_target_volume,
            contract,
            surface_factory=_MonotoneVolumeSurfaceFactory,
        )

        names = [config["name"] for config in configs]
        seed_labels = [config["seed_label"] for config in configs]
        target_volumes = [config["target_volume"] for config in configs]

        # The real builder produced the 4-surface interior-covering stack,
        # inner-to-outer, with strictly increasing volumes and the inner shell
        # reaching the configured 0.4 label fraction toward the core.
        self.assertEqual(names, ["inner0", "inner1", "inner2", "outer"])
        self.assertTrue(
            all(
                left < right
                for left, right in zip(target_volumes[:-1], target_volumes[1:])
            )
        )
        self.assertAlmostEqual(seed_labels[0], seed_label * 0.4)
        self.assertAlmostEqual(seed_labels[-1], seed_label)
        self.assertAlmostEqual(target_volumes[-1], outer_target_volume)

    def test_build_surface_configs_for_contract_default_published_is_unchanged(self):
        contracts_module = load_surface_mode_contracts_module()
        geometry_module = load_single_stage_geometry_module()

        contract = contracts_module.build_surface_mode_contract(
            requested_surface_mode=contracts_module.PUBLISHED_MULTISURFACE,
            legacy_num_surfaces=1,
            legacy_inner_surface_ratio=0.8,
        )

        configs = geometry_module.build_surface_configs_for_contract(
            "dummy.nc",
            8,
            8,
            0.9,
            1.0,
            0.5,
            contract,
            surface_factory=_MonotoneVolumeSurfaceFactory,
        )

        self.assertEqual(
            [config["name"] for config in configs],
            ["inner0", "inner1", "outer"],
        )
        self.assertEqual(
            [config["seed_label"] for config in configs],
            [0.9 * 0.6, 0.9 * 0.8, 0.9 * 1.0],
        )


class SingleStageCliSurfaceDepthTests(unittest.TestCase):
    """CLI wiring for the opt-in published-stack depth knobs."""

    def _published_cli_args(self, **overrides):
        # "published_multisurface" / "default_v1" are the SSOT string constants
        # exercised against the loaded module's PUBLISHED_MULTISURFACE /
        # PUBLISHED_PRESET_DEFAULT_V1 below.
        values = {
            "surface_mode": "published_multisurface",
            "num_surfaces": 1,
            "inner_surface_ratio": 0.8,
            "published_surface_preset": "default_v1",
            "published_surface_fractions": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_default_cli_resolves_byte_identical_published_stack(self):
        module = load_single_stage_example_module()
        # Guard the string literals used by these CLI tests against the SSOT.
        self.assertEqual(module.PUBLISHED_MULTISURFACE, "published_multisurface")
        self.assertEqual(module.PUBLISHED_PRESET_DEFAULT_V1, "default_v1")
        args = self._published_cli_args()

        contract = module.resolve_surface_mode_contract(
            args, warn_on_legacy_mapping=False
        )

        self.assertEqual(contract.label_fractions, (0.6, 0.8, 1.0))
        self.assertEqual(contract.num_surfaces, 3)

    def test_cli_preset_selects_interior_covering_stack(self):
        module = load_single_stage_example_module()
        args = self._published_cli_args(
            published_surface_preset="interior_covering",
        )

        contract = module.resolve_surface_mode_contract(
            args, warn_on_legacy_mapping=False
        )

        self.assertEqual(contract.label_fractions, (0.4, 0.6, 0.8, 1.0))
        self.assertEqual(contract.num_surfaces, 4)

    def test_cli_explicit_fractions_string_is_parsed_and_validated(self):
        module = load_single_stage_example_module()
        args = self._published_cli_args(
            published_surface_fractions="0.3, 0.55, 0.8, 1.0",
        )

        contract = module.resolve_surface_mode_contract(
            args, warn_on_legacy_mapping=False
        )

        self.assertEqual(contract.label_fractions, (0.3, 0.55, 0.8, 1.0))

    def test_cli_explicit_fractions_string_fail_closed_on_bad_stack(self):
        module = load_single_stage_example_module()
        args = self._published_cli_args(
            published_surface_fractions="0.8, 0.6, 1.0",
        )

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            module.resolve_surface_mode_contract(args, warn_on_legacy_mapping=False)

    def test_cli_empty_fractions_string_rejected(self):
        module = load_single_stage_example_module()

        with self.assertRaisesRegex(ValueError, "at least one label fraction"):
            module.parse_published_surface_fractions(" , ,")

    def test_cli_default_single_surface_run_is_byte_identical(self):
        module = load_single_stage_example_module()
        # A default single-surface run carries the default-v1 preset value but
        # never sets fractions; the depth knob must be inert for non-published
        # modes (default preset is allowed; it just resolves to the baseline).
        args = SimpleNamespace(
            surface_mode=None,
            num_surfaces=1,
            inner_surface_ratio=0.8,
            published_surface_preset="default_v1",
            published_surface_fractions=None,
        )

        contract = module.resolve_surface_mode_contract(
            args, warn_on_legacy_mapping=False
        )

        self.assertEqual(contract.mode, module.SINGLE_SURFACE)
        self.assertEqual(contract.label_fractions, (1.0,))


if __name__ == "__main__":
    unittest.main()
