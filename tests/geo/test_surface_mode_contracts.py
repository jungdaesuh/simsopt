import importlib.util
import sys
import unittest
import uuid
import warnings
from pathlib import Path
from types import SimpleNamespace


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
SURFACE_MODE_CONTRACTS_PATH = (
    EXAMPLE_ROOT / "banana_opt" / "surface_mode_contracts.py"
)
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
    return load_module(SINGLE_STAGE_ENTRYPOINT_PATH, "single_stage_banana_example")


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
        self.assertEqual(contract.source, module.SURFACE_MODE_SOURCE_LEGACY_NUM_SURFACES_MAPPING)
        self.assertEqual(contract.label_fractions, (0.7, 1.0))
        self.assertEqual(contract.weights, (1.0, 1.0))
        self.assertEqual(contract.legacy_num_surfaces, 2)
        self.assertEqual(contract.legacy_inner_surface_ratio, 0.7)
        self.assertTrue(any(issubclass(entry.category, DeprecationWarning) for entry in caught))

    def test_explicit_surface_mode_uses_effective_contract_and_clears_legacy_fields(self):
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

    def test_published_multisurface_contract_has_fixed_stack_and_is_runtime_rejected(self):
        module = load_surface_mode_contracts_module()

        contract = module.build_surface_mode_contract(
            requested_surface_mode=module.PUBLISHED_MULTISURFACE,
            legacy_num_surfaces=1,
            legacy_inner_surface_ratio=0.8,
        )

        self.assertEqual(contract.label_fractions, (0.6, 0.8, 1.0))
        self.assertEqual(contract.weights, (1.0, 1.0, 1.0))
        self.assertEqual(contract.stack_policy, module.SURFACE_STACK_POLICY_PUBLISHED_FIXED_STACK)
        with self.assertRaisesRegex(ValueError, "not implemented yet"):
            module.validate_surface_mode_runtime_support(contract)


class SingleStageSurfaceModeIntegrationTests(unittest.TestCase):
    def test_resolve_surface_mode_contract_suppresses_legacy_warning_for_default_single_surface(self):
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
            topology_scorer_every=0,
            topology_scorer_nfieldlines=12,
            topology_scorer_tmax=50.0,
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
        contract = module.resolve_surface_mode_contract(args, warn_on_legacy_mapping=False)

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
        self.assertEqual(config.inner_surface_ratio, 0.73)

    def test_validate_boozer_stage_refinement_args_rejects_explicit_multisurface_contract(self):
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

    def test_validate_surface_mode_constraint_args_rejects_explicit_multisurface_alm(self):
        module = load_single_stage_example_module()
        args, contract = resolve_explicit_multisurface_contract(
            module,
            constraint_method="alm",
        )

        with self.assertRaisesRegex(
            ValueError,
            f"--surface-mode={module.SINGLE_SURFACE}",
        ):
            module.validate_surface_mode_constraint_args(
                args,
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


if __name__ == "__main__":
    unittest.main()
