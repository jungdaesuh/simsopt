import importlib
import importlib.util
import sys
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
FRONTIER_CAMPAIGN_PATH = EXAMPLE_ROOT / "run_single_stage_frontier_campaign.py"
GOAL_MODE_COMPARISON_PATH = EXAMPLE_ROOT / "run_single_stage_goal_mode_comparison.py"


def ensure_examples_import_path() -> None:
    examples_root = str(EXAMPLE_ROOT)
    if examples_root not in sys.path:
        sys.path.insert(0, examples_root)


ensure_examples_import_path()


def load_module(path: Path, stem: str):
    spec = importlib.util.spec_from_file_location(f"{stem}_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_banana_opt_module(module_name: str):
    return importlib.import_module(f"banana_opt.{module_name}")


def load_frontier_archive_module(): return load_banana_opt_module("frontier_archive")


def load_frontier_campaign_module(): return load_module(FRONTIER_CAMPAIGN_PATH, "run_single_stage_frontier_campaign")


def load_frontier_campaign_execution_module(): return load_banana_opt_module("frontier_campaign_execution")


def load_frontier_conditioning_module(): return load_banana_opt_module("frontier_conditioning")


def load_frontier_constraints_module(): return load_banana_opt_module("single_stage_search_contracts")


def load_frontier_contracts_module(): return load_banana_opt_module("frontier_contracts")


def load_frontier_dominance_module(): return load_banana_opt_module("frontier_dominance")


def load_frontier_progress_state_module(): return load_banana_opt_module("frontier_progress_state")


def load_frontier_recommendation_module(): return load_banana_opt_module("frontier_recommendation")


def load_frontier_reporting_module(): return load_banana_opt_module("frontier_campaign_reporting")


def load_frontier_runtime_calibration_module(): return load_banana_opt_module("frontier_runtime_calibration")


def load_frontier_scalarization_module(): return load_banana_opt_module("frontier_scalarization")


def load_frontier_solver_checkpoint_module(): return load_banana_opt_module("frontier_solver_checkpoint")


def load_goal_mode_comparison_module():
    return load_module(
        GOAL_MODE_COMPARISON_PATH,
        "run_single_stage_goal_mode_comparison",
    )


def load_incumbents_module(): return load_banana_opt_module("incumbents")


def load_search_evaluation_module(): return load_banana_opt_module("search_evaluation")


def load_search_policy_module(): return load_banana_opt_module("single_stage_search_policy")


def make_frontier_archive_member(
    archive_module,
    *,
    member_id: str,
    iota: float,
    volume: float,
    qa_error: float,
    boozer_residual: float,
    soft_search_score: float = -1.0,
    frontier_trust_ok: bool | None = True,
    hardware_constraints_ok: bool | None = True,
    archive_state: str = "certified",
    hard_certification_ok: bool = True,
    reference_metrics: dict[str, float] | None = None,
    results_path: str | None = None,
):
    constraint_metrics: dict[str, object] = {}
    if frontier_trust_ok is not None:
        constraint_metrics["frontier_trust_ok"] = frontier_trust_ok
    if hardware_constraints_ok is not None:
        constraint_metrics["hardware_constraints_ok"] = hardware_constraints_ok
    if hard_certification_ok:
        constraint_metrics["frontier_certification_ok"] = True
        constraint_metrics["frontier_certification_reason"] = "certified"
        constraint_metrics["frontier_invariant_torus_fraction"] = 1.0
        constraint_metrics["frontier_invariant_torus_min"] = 0.0
        constraint_metrics["frontier_kam_fraction"] = 1.0
        constraint_metrics["frontier_kam_min"] = 0.0
    return archive_module.FrontierArchiveMember(
        member_id=member_id,
        lane_id=member_id.split(":")[-1],
        campaign_id="campaign",
        archive_state=archive_state,
        dominance_signature={},
        objective_metrics={
            "iota": iota,
            "volume": volume,
            "qa_error": qa_error,
            "boozer_residual": boozer_residual,
        },
        reference_metrics=(
            {
                "iota": 0.15,
                "volume": 0.10,
                "qa_error": 0.012,
                "boozer_residual": 0.008,
            }
            if reference_metrics is None
            else dict(reference_metrics)
        ),
        constraint_metrics=constraint_metrics,
        hard_certification_ok=hard_certification_ok,
        soft_search_score=soft_search_score,
        hypervolume_contribution=None,
        recommendation_flags={},
        rerun_contract={},
        result_source="final",
        results_path=f"/tmp/{member_id}.json" if results_path is None else results_path,
        termination_reason="ok",
        success=True,
    )
