from __future__ import annotations

import ast
import copy
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import benchmarks.run_single_stage_changed_state_gpu_timeline as timeline_runner
import benchmarks.summarize_single_stage_changed_state_gpu_timeline as timeline_summary
import numpy as np
import pytest
from benchmarks.run_jax_native_example_measurements import execute_monitored_command
from benchmarks.run_single_stage_changed_state_gpu_timeline import (
    ARTIFACT_SCHEMA_ID,
    CASE_ID,
    TimelineRunnerError,
    build_child_command,
    child_schedule,
    validate_artifact_root,
)
from benchmarks.single_stage_changed_state_trace_preflight import (
    EXPECTED_DEVICE_PROCESS,
    OBSERVED_EVIDENCE_COUNT_FIELDS,
    PREFLIGHT_SCHEMA_ID,
)
from benchmarks.summarize_single_stage_changed_state_gpu_timeline import (
    Interval,
    TraceSummaryError,
    summarize_trace_document,
    union_intervals,
    unique_deepest_scope,
)
from examples.jax.parity.cases import native_boozerqa
from examples.jax.parity.input_bundle import create_input_bundle
from simsopt_jax.runtime.trace_annotations import (
    EvaluationKind,
    EvaluationTraceContext,
    HostEvent,
    HostEventRecord,
    PhaseId,
    accepted_iteration_span,
)

_HOST_PID = 701
_DEVICE_PID = 1


class TestBoundedTimelineRunner:
    def test_schedule_is_exactly_three_alternating_fresh_pairs(self) -> None:
        schedule = child_schedule()

        assert [entry.child_id for entry in schedule] == [
            "profiled-0",
            "control-0",
            "profiled-1",
            "control-1",
            "profiled-2",
            "control-2",
        ]
        assert [entry.order_index for entry in schedule] == list(range(6))
        assert [entry.pair_index for entry in schedule] == [0, 0, 1, 1, 2, 2]
        assert len({entry.child_id for entry in schedule}) == 6

    @pytest.mark.parametrize(
        ("profile_children", "control_children"), ((2, 3), (3, 2), (4, 4))
    )
    def test_schedule_rejects_protocol_count_changes(
        self, profile_children: int, control_children: int
    ) -> None:
        with pytest.raises(ValueError, match="exactly three"):
            child_schedule(profile_children, control_children)

    def test_artifact_root_is_fresh_non_tmp_and_diagnostic_specific(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        valid = Path.home() / ARTIFACT_SCHEMA_ID
        assert validate_artifact_root(valid) == valid.resolve()

        with pytest.raises(TimelineRunnerError, match="identify"):
            validate_artifact_root(Path.home() / "single-stage-speed-20260804")
        with pytest.raises(TimelineRunnerError, match="/tmp"):
            validate_artifact_root(Path("/tmp") / ARTIFACT_SCHEMA_ID)
        with pytest.raises(TimelineRunnerError, match="outside the repo"):
            validate_artifact_root(
                Path(__file__).resolve().parents[2] / ARTIFACT_SCHEMA_ID
            )
        monkeypatch.setattr(Path, "exists", lambda path: path == valid)
        with pytest.raises(FileExistsError, match="already exists"):
            validate_artifact_root(valid)

    def test_child_command_uses_the_same_static_runner_in_a_fresh_process(
        self, tmp_path: Path
    ) -> None:
        command = build_child_command(sys.executable, tmp_path / "profiled-0.json")

        assert command[0] == sys.executable
        assert Path(command[1]).name == (
            "run_single_stage_changed_state_gpu_timeline.py"
        )
        assert command[2:] == (
            "--child-spec",
            str((tmp_path / "profiled-0.json").resolve()),
        )

    def test_runner_does_not_import_or_extend_the_frozen_r5_artifact_modules(
        self,
    ) -> None:
        runner_path = (
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "run_single_stage_changed_state_gpu_timeline.py"
        )
        tree = ast.parse(runner_path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }

        assert CASE_ID == "native-single-stage-boozer-vacuum-optimization"
        assert "benchmarks.single_stage_speed_campaign_receipt" not in imported_modules
        assert "benchmarks.validate_single_stage_speed_claim" not in imported_modules

    def test_process_tree_rss_bound_terminates_the_fresh_child(
        self, tmp_path: Path
    ) -> None:
        result = execute_monitored_command(
            command=(sys.executable, "-c", "import time; time.sleep(30)"),
            environment={},
            cwd=tmp_path,
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
            device="cpu",
            gpu_index=0,
            poll_interval_seconds=0.001,
            timeout_seconds=5.0,
            max_process_tree_rss_bytes=1,
        )

        assert result.returncode != 0
        assert result.termination == "process_tree_memory_limit"
        assert result.peak_process_tree_rss_bytes > 1

    def test_process_tree_rss_bound_must_be_positive(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="RSS bound"):
            execute_monitored_command(
                command=(sys.executable, "-c", "pass"),
                environment={},
                cwd=tmp_path,
                stdout_path=tmp_path / "stdout.log",
                stderr_path=tmp_path / "stderr.log",
                device="cpu",
                gpu_index=0,
                poll_interval_seconds=0.01,
                timeout_seconds=1.0,
                max_process_tree_rss_bytes=0,
            )

    def test_wall_bound_terminates_the_fresh_child(self, tmp_path: Path) -> None:
        result = execute_monitored_command(
            command=(sys.executable, "-c", "import time; time.sleep(30)"),
            environment={},
            cwd=tmp_path,
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
            device="cpu",
            gpu_index=0,
            poll_interval_seconds=0.001,
            timeout_seconds=0.02,
            max_process_tree_rss_bytes=1024**3,
        )

        assert result.returncode != 0
        assert result.termination == "wall_time_limit"

    def test_route_is_the_exact_production_direct_adjoint_route(self) -> None:
        assert timeline_runner._route_document() == {
            "optimizer": "SIMSOPT_LBFGSB",
            "driver": "minimize_lbfgs_host_core",
            "line_search": "line_search_value_and_grad_host",
            "adjoint_route": "exact_jacobian_dense_fp64_lu",
        }

    def test_claimed_environment_excludes_child_cache_paths(self) -> None:
        environment = {
            "JAX_ENABLE_X64": "1",
            "SIMSOPT_EXACT_ADJOINT_DENSE_LU": "1",
            "TF_PROFILER_TRACE_VIEWER_MAX_EVENTS": "67108864",
            "JAX_COMPILATION_CACHE_DIR": "/cache/profiled-0",
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        }

        claimed = timeline_runner._claimed_environment(environment)

        assert claimed == {
            "JAX_ENABLE_X64": "1",
            "SIMSOPT_EXACT_ADJOINT_DENSE_LU": "1",
            "TF_PROFILER_TRACE_VIEWER_MAX_EVENTS": "67108864",
        }
        assert timeline_runner._sha256_bytes(
            timeline_runner.canonical_json_bytes(claimed)
        ) == timeline_runner._sha256_bytes(
            timeline_runner.canonical_json_bytes(
                timeline_runner._claimed_environment(
                    {
                        **environment,
                        "JAX_COMPILATION_CACHE_DIR": "/cache/control-2",
                    }
                )
            )
        )

    def test_profiled_policy_materializes_exact_jax_options(self) -> None:
        class FakeProfileOptions:
            def __init__(self) -> None:
                self.host_tracer_level = -1
                self.python_tracer_level = -1
                self.advanced_configuration: dict[str, int] = {}

        fake_jax = SimpleNamespace(
            profiler=SimpleNamespace(ProfileOptions=FakeProfileOptions)
        )
        policy = timeline_runner._profiler_policy("profiled")

        options = timeline_runner._jax_profiler_options(fake_jax, policy)

        assert timeline_runner._profiler_policy_document(policy) == {
            "enabled": True,
            "host_tracer_level": 1,
            "python_tracer_level": 0,
            "device_tracing": "jax_default",
            "trace_viewer_max_events": 67_108_864,
            "advanced_configuration": {
                "gpu_max_activity_api_events": 33_554_432,
                "gpu_max_callback_api_events": 33_554_432,
            },
        }
        assert options.host_tracer_level == 1
        assert options.python_tracer_level == 0
        assert not hasattr(options, "device_tracer_level")
        assert options.advanced_configuration == {
            "gpu_max_activity_api_events": 33_554_432,
            "gpu_max_callback_api_events": 33_554_432,
        }

    def test_control_policy_is_explicitly_unprofiled(self) -> None:
        policy = timeline_runner._profiler_policy("control")

        assert timeline_runner._profiler_policy_document(policy) == {
            "enabled": False,
            "host_tracer_level": None,
            "python_tracer_level": None,
            "device_tracing": None,
            "trace_viewer_max_events": None,
            "advanced_configuration": {},
        }
        with pytest.raises(TimelineRunnerError, match="unexpected profiler policy"):
            timeline_runner._jax_profiler_options(
                SimpleNamespace(
                    profiler=SimpleNamespace(ProfileOptions=lambda: object())
                ),
                policy,
            )

    def test_profiler_policy_parser_rejects_capacity_tamper(self) -> None:
        policy = timeline_runner._profiler_policy_document(
            timeline_runner._profiler_policy("profiled")
        )
        advanced = policy["advanced_configuration"]
        assert isinstance(advanced, dict)
        advanced["gpu_max_activity_api_events"] = 1_000_000

        with pytest.raises(TimelineRunnerError, match="differs from schema"):
            timeline_runner._parse_profiler_policy(policy)

    @pytest.mark.parametrize("mode", ("profiled", "control"))
    def test_child_spec_round_trip_binds_mode_specific_profiler_policy(
        self, tmp_path: Path, mode: str
    ) -> None:
        entry = next(item for item in child_schedule() if item.mode == mode)
        spec = timeline_runner._child_spec(
            entry,
            workspace=tmp_path,
            source_state_sha256="a" * 64,
            environment_sha256="b" * 64,
            input_sha256="c" * 64,
            configuration_sha256="d" * 64,
            construction_sha256="e" * 64,
            runtime_policy_sha256="f" * 64,
            initial_parameters_sha256="1" * 64,
            device_name="gpu",
            device_uuid="GPU-1",
            environment={"JAX_ENABLE_X64": "1"},
            phase_ids=("outer.iteration",),
            trace_schema_id="trace-v1",
        )
        spec_path = tmp_path / f"{mode}.json"
        timeline_runner._write_json_exclusive(
            spec_path, timeline_runner._child_spec_document(spec)
        )

        restored = timeline_runner._read_child_spec(spec_path)

        assert restored == spec
        assert restored.profiler_policy == timeline_runner._profiler_policy(mode)

    def test_profiler_retention_binds_exact_cupti_drop_warning(self) -> None:
        warning = b"Already too many activity events, drop the buffer"

        assert timeline_runner._profiler_retention_document("profiled", b"") == {
            "evidence_available": True,
            "activity_buffers_dropped": False,
            "warning": None,
        }
        assert timeline_runner._profiler_retention_document(
            "profiled", b"prefix " + warning + b" suffix"
        ) == {
            "evidence_available": True,
            "activity_buffers_dropped": True,
            "warning": warning.decode("utf-8"),
        }
        assert timeline_runner._profiler_retention_document("control", warning) == {
            "evidence_available": False,
            "activity_buffers_dropped": None,
            "warning": None,
        }

    def test_segmented_boundary_records_require_exactly_seven_ordered_pairs(
        self,
    ) -> None:
        from simsopt_jax.runtime.trace_annotations import ProfilerBoundaryOperation

        records = tuple(
            SimpleNamespace(
                iteration_id=iteration_id,
                operation=operation,
                start_ns=iteration_id * 100 + offset,
                end_ns=iteration_id * 100 + offset + 5,
            )
            for iteration_id in range(1, 8)
            for operation, offset in (
                (ProfilerBoundaryOperation.START, 0),
                (ProfilerBoundaryOperation.STOP, 10),
            )
        )

        timeline_runner._validate_complete_boundary_records(records)
        documents = timeline_runner._boundary_pause_documents(records)

        assert len(documents) == 14
        assert sum(int(row["duration_ns"]) for row in documents) == 70
        assert documents[0]["operation"] == "start"
        assert documents[-1]["operation"] == "stop"
        with pytest.raises(
            timeline_runner.TraceCollectionError,
            match="exact sequential",
        ):
            timeline_runner._validate_complete_boundary_records(records[:-1])

    def test_segmented_measurement_runs_one_optimizer_execution_with_seven_sessions(
        self,
    ) -> None:
        calls: list[str] = []
        execution_count = 0

        def execute() -> str:
            nonlocal execution_count
            execution_count += 1
            for iteration_id in range(1, 8):
                with accepted_iteration_span(iteration_id):
                    calls.append(f"step-{iteration_id}")
            return "unchanged-result"

        result, _audit, boundary_audit = (
            timeline_runner._execute_segmented_profiled_measurement(
                execute,
                start_segment=lambda iteration_id: calls.append(
                    f"start-{iteration_id}"
                ),
                stop_segment=lambda iteration_id: calls.append(f"stop-{iteration_id}"),
            )
        )

        assert result == "unchanged-result"
        assert execution_count == 1
        assert calls == [
            item
            for iteration_id in range(1, 8)
            for item in (
                f"start-{iteration_id}",
                f"step-{iteration_id}",
                f"stop-{iteration_id}",
            )
        ]
        timeline_runner._validate_complete_boundary_records(boundary_audit.records())

    def test_segment_digest_binds_rejected_and_accepted_trials(self) -> None:
        evaluations = (
            {"evaluation_id": "initial", "lifecycle": "initial", "iteration": None},
            {"evaluation_id": "trial-rejected", "lifecycle": "trial", "iteration": 1},
            {"evaluation_id": "trial-accepted", "lifecycle": "trial", "iteration": 1},
            {"evaluation_id": "trial-next", "lifecycle": "trial", "iteration": 2},
        )

        assert timeline_runner._segment_evaluation_ids(evaluations, 1) == (
            "trial-rejected",
            "trial-accepted",
        )
        with pytest.raises(TimelineRunnerError, match="no trial evaluation"):
            timeline_runner._segment_evaluation_ids(evaluations, 7)

    def test_segment_pause_attribution_follows_containing_target_gap(self) -> None:
        from simsopt_jax.runtime.trace_annotations import ProfilerBoundaryOperation

        records = (
            SimpleNamespace(
                iteration_id=1,
                operation=ProfilerBoundaryOperation.STOP,
                start_ns=110,
                end_ns=120,
            ),
            SimpleNamespace(
                iteration_id=2,
                operation=ProfilerBoundaryOperation.START,
                start_ns=130,
                end_ns=140,
            ),
            SimpleNamespace(
                iteration_id=7,
                operation=ProfilerBoundaryOperation.STOP,
                start_ns=300,
                end_ns=310,
            ),
        )
        gaps = (
            SimpleNamespace(
                phase=PhaseId.HOST_LINE_SEARCH_CONTROL,
                start_ns=100,
                end_ns=150,
                attributes=(("outer_iteration_id", 2),),
            ),
        )

        pauses = timeline_runner._iteration_pause_intervals(records, gaps, 2)

        assert [(pause.start_ns, pause.end_ns) for pause in pauses] == [
            (110, 120),
            (130, 140),
        ]

    def test_source_state_drift_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            timeline_runner, "_timeline_source_state_sha256", lambda _root: "b" * 64
        )

        with pytest.raises(TimelineRunnerError, match="source state changed"):
            timeline_runner._assert_source_state("a" * 64)

    def test_source_state_hash_covers_untracked_source_bytes(
        self, tmp_path: Path
    ) -> None:
        subprocess.run(("git", "init", "-q"), cwd=tmp_path, check=True)
        tracked = tmp_path / "tracked.py"
        tracked.write_text("TRACKED = True\n", encoding="utf-8")
        subprocess.run(("git", "add", "tracked.py"), cwd=tmp_path, check=True)
        subprocess.run(
            (
                "git",
                "-c",
                "user.name=Timeline Test",
                "-c",
                "user.email=timeline@example.invalid",
                "commit",
                "-qm",
                "seed",
            ),
            cwd=tmp_path,
            check=True,
        )
        source = tmp_path / "new_runner.py"
        source.write_text("VALUE = 1\n", encoding="utf-8")
        before = timeline_runner._timeline_source_state_sha256(tmp_path)

        source.write_text("VALUE = 2\n", encoding="utf-8")

        assert timeline_runner._timeline_source_state_sha256(tmp_path) != before

    def test_identity_preimages_recompute_all_parent_frozen_hashes(
        self, tmp_path: Path
    ) -> None:
        arrays = {
            "coil_dofs": np.asarray([1.0, -2.0], dtype=np.float64),
            "surface_dofs": np.asarray([3.0], dtype=np.float64),
        }
        bundle = create_input_bundle(
            tmp_path,
            case_id=CASE_ID,
            random_seed=7,
            arrays=arrays,
            configuration={"mpol": 3, "ntor": 2},
            scale="native_default",
        )
        runtime_policy = {
            "route": timeline_runner._route_document(),
            "accepted_iterations": 7,
        }

        document = timeline_runner._identity_preimages_document(
            bundle,
            arrays,
            runtime_policy,
            SimpleNamespace(
                simsoptpp_path="/opt/simsoptpp.so",
                simsoptpp_sha256="a" * 64,
                simsoptpp_build_commit="b" * 40,
            ),
            (
                {
                    "original_path": "runner.py",
                    "manifest_path": f"source_preimages/{'c' * 64}",
                    "sha256": "c" * 64,
                    "blob_id": "c" * 64,
                },
            ),
        )

        assert (
            timeline_runner._sha256_bytes(
                timeline_runner.canonical_json_bytes(
                    document["input_fingerprint_payload"]
                )
            )
            == bundle.input_fingerprint
        )
        assert (
            timeline_runner._sha256_bytes(
                timeline_runner.canonical_json_bytes(document["configuration"])
            )
            == bundle.configuration_fingerprint
        )
        assert timeline_runner._sha256_bytes(
            timeline_runner.canonical_json_bytes(
                document["construction_fingerprint_payload"]
            )
        ) == timeline_runner._construction_sha256(bundle, arrays)
        assert timeline_runner._sha256_bytes(
            timeline_runner.canonical_json_bytes(document["runtime_policy_payload"])
        ) == timeline_runner._sha256_bytes(
            timeline_runner.canonical_json_bytes(runtime_policy)
        )
        assert document["simsoptpp"] == {
            "path": "/opt/simsoptpp.so",
            "sha256": "a" * 64,
            "build_commit": "b" * 40,
        }
        assert document["source_preimages"] == [
            {
                "original_path": "runner.py",
                "manifest_path": f"source_preimages/{'c' * 64}",
                "sha256": "c" * 64,
                "blob_id": "c" * 64,
            }
        ]

    def test_preflight_acceptance_cross_binds_schema_scopes_and_device(self) -> None:
        evidence = {
            "schema_id": PREFLIGHT_SCHEMA_ID,
            "state": "pass",
            "trace_schema_id": "trace-v1",
            "required_scopes": ["newton.residual_jvp", "adjoint.lu_solve"],
            "observed_evidence": [
                {
                    "phase_id": phase_id,
                    OBSERVED_EVIDENCE_COUNT_FIELDS[0]: 1,
                    OBSERVED_EVIDENCE_COUNT_FIELDS[1]: 1,
                    OBSERVED_EVIDENCE_COUNT_FIELDS[2]: 0,
                }
                for phase_id in ("newton.residual_jvp", "adjoint.lu_solve")
            ],
            "device_identity": {"name": "gpu", "uuid": "GPU-1"},
            "profiler_policy": timeline_runner._profiler_policy_document(
                timeline_runner._profiler_policy("profiled")
            ),
            "session_evidence": [
                {
                    "session_id": f"session-{index:02d}",
                    "device_processes": [EXPECTED_DEVICE_PROCESS],
                    "observed_evidence": [
                        {
                            "phase_id": phase_id,
                            OBSERVED_EVIDENCE_COUNT_FIELDS[0]: 1,
                            OBSERVED_EVIDENCE_COUNT_FIELDS[1]: 1,
                            OBSERVED_EVIDENCE_COUNT_FIELDS[2]: 0,
                        }
                        for phase_id in (
                            "newton.residual_jvp",
                            "adjoint.lu_solve",
                        )
                    ],
                }
                for index in (1, 2)
            ],
            "failure_reason": None,
        }

        timeline_runner._validate_preflight_evidence(
            evidence,
            trace_schema_id="trace-v1",
            device_name="gpu",
            device_uuid="GPU-1",
        )

        for field, changed in (
            ("trace_schema_id", "trace-v2"),
            ("required_scopes", ["newton.residual_jvp"]),
            ("device_identity", {"name": "other", "uuid": "GPU-1"}),
            (
                "profiler_policy",
                {
                    **evidence["profiler_policy"],
                    "host_tracer_level": 2,
                },
            ),
        ):
            invalid = {**evidence, field: changed}
            with pytest.raises(TimelineRunnerError, match="preflight"):
                timeline_runner._validate_preflight_evidence(
                    invalid,
                    trace_schema_id="trace-v1",
                    device_name="gpu",
                    device_uuid="GPU-1",
                )

    def test_null_simsoptpp_build_commit_is_diagnostic_only(self) -> None:
        diagnostic = SimpleNamespace(
            executed_sources=(object(),),
            simsoptpp_path="/opt/simsoptpp.so",
            simsoptpp_sha256="a" * 64,
            simsoptpp_build_commit=None,
            authoritative=False,
        )
        timeline_runner._require_runtime_provenance(
            diagnostic, context="parent prelaunch"
        )

        with pytest.raises(TimelineRunnerError, match="authoritative provenance"):
            timeline_runner._require_runtime_provenance(
                SimpleNamespace(**{**vars(diagnostic), "authoritative": True}),
                context="parent prelaunch",
            )

    def test_unavailable_newton_trace_preserves_real_iteration_count(self) -> None:
        unavailable = {
            "newton_iterations": 3,
            "newton_attempted_iterations": None,
            "newton_trace_available": False,
            "residual_trace": [],
            "step_accepted_trace": [],
            "linear_solve_success_trace": [],
        }

        timeline_runner._validate_newton_inner_evidence(unavailable)

        for changed in (
            {**unavailable, "newton_attempted_iterations": 0},
            {**unavailable, "step_accepted_trace": [True]},
            {**unavailable, "newton_trace_available": True},
            {**unavailable, "newton_iterations": -1},
        ):
            with pytest.raises(TimelineRunnerError, match="Newton"):
                timeline_runner._validate_newton_inner_evidence(changed)

    def test_available_newton_trace_requires_exact_typed_consistency(self) -> None:
        available = {
            "newton_iterations": 2,
            "newton_attempted_iterations": 3,
            "newton_trace_available": True,
            "residual_trace": [1.0e-2, 1.0e-4, 1.0e-8],
            "step_accepted_trace": [True, False, True],
            "linear_solve_success_trace": [True, True, True],
        }

        timeline_runner._validate_newton_inner_evidence(available)

        for changed in (
            {**available, "newton_attempted_iterations": 2},
            {**available, "step_accepted_trace": [1, False, True]},
            {**available, "residual_trace": [True, 1.0e-4, 1.0e-8]},
            {**available, "newton_iterations": 3},
        ):
            with pytest.raises(TimelineRunnerError, match="Newton"):
                timeline_runner._validate_newton_inner_evidence(changed)

    def test_prepared_runtime_reuses_identity_but_mints_fresh_controller(self) -> None:
        class Session:
            pass

        incumbent_evaluator = object()

        runtime = native_boozerqa._PreparedJaxRuntime(
            session=Session(),
            runtime={},
            reporting=dict,
            value_and_grad=lambda values: values,
            initial_parameters=np.asarray([1.0]),
            initial_inner_success=True,
            iota_target=-0.4,
            initial_volume=0.1,
            optimizer_backend=None,
            exact_newton_variant="C0",
            incumbent_evaluator=incumbent_evaluator,
            incumbent_factory=object,
        )
        identity = (
            id(runtime.session),
            id(runtime.runtime),
            id(runtime.value_and_grad),
            id(runtime.incumbent_evaluator),
        )

        first = runtime.fresh_incumbent_controller()
        second = runtime.fresh_incumbent_controller()

        assert first is not second
        assert (
            id(runtime.session),
            id(runtime.runtime),
            id(runtime.value_and_grad),
            id(runtime.incumbent_evaluator),
        ) == identity

    def test_frozen_r5_root_cannot_be_used_as_timeline_root(self) -> None:
        with pytest.raises(TimelineRunnerError, match="frozen r5 root"):
            validate_artifact_root(Path.home() / "campaign-20260804-frozen-r5")
        with pytest.raises(TimelineRunnerError, match="frozen r5 root"):
            validate_artifact_root(
                Path.home() / "campaign-20260804-frozen-r5" / ARTIFACT_SCHEMA_ID
            )


def _metadata_event(pid: int, name: str) -> dict[str, object]:
    return {"ph": "M", "pid": pid, "name": "process_name", "args": {"name": name}}


def _span(
    name: str,
    ts_us: float,
    dur_us: float,
    *,
    pid: int = _HOST_PID,
    args: dict[str, object] | None = None,
) -> dict[str, object]:
    normalized_args = {} if args is None else dict(args)
    if pid == _DEVICE_PID and "kernel_details" in normalized_args:
        normalized_args = {
            "context_id": "$$1",
            "correlation_id": "1",
            "hlo_module": "jit_f",
            "hlo_op": name,
            "scope_range_id": name,
            "tf_op": "XlaModule:",
            **normalized_args,
        }
    if pid == _DEVICE_PID and "memcpy_details" in normalized_args:
        normalized_args = {
            "context_id": "$$1",
            "correlation_id": "1",
            **normalized_args,
        }
    return {
        "ph": "X",
        "pid": pid,
        "tid": 1,
        "ts": ts_us,
        "dur": dur_us,
        "name": name,
        "args": normalized_args,
    }


def _lifecycle_args(
    evaluation_id: str,
    parameter_sha256: str,
    kind: EvaluationKind,
    iteration: int | None,
) -> dict[str, object]:
    args: dict[str, object] = {
        "evaluation_id": evaluation_id,
        "evaluation_kind": kind.value,
        "parameter_sha256": parameter_sha256,
    }
    if iteration is not None:
        args["outer_iteration_id"] = str(iteration)
    return args


def _host_evaluation_events(
    *,
    timestamps_ns: tuple[int, int, int],
    evaluation_id: str,
    parameter_sha256: str,
    kind: EvaluationKind,
    iteration: int | None,
) -> tuple[HostEventRecord, ...]:
    context = EvaluationTraceContext(
        evaluation_id=evaluation_id,
        parameter_sha256=parameter_sha256,
        kind=kind,
        outer_iteration_id=iteration,
    )
    return tuple(
        HostEventRecord(
            sequence=index,
            event=event,
            timestamp_ns=timestamps_ns[index],
            evaluation=context,
            attributes=(),
        )
        for index, event in enumerate(HostEvent)
    )


def _fixture() -> tuple[dict[str, object], tuple[HostEventRecord, ...]]:
    initial_args = _lifecycle_args("initial", "a" * 64, EvaluationKind.INITIAL, None)
    trial_args = _lifecycle_args("trial-1", "b" * 64, EvaluationKind.TRIAL, 1)
    events: list[dict[str, object]] = [
        _metadata_event(_DEVICE_PID, "/device:GPU:0"),
        _metadata_event(_HOST_PID, "/host:CPU"),
        _span("optimizer.lifecycle.evaluator_entry", 800.0, 0.001, args=initial_args),
        _span("optimizer.lifecycle.device_ready", 900.0, 0.001, args=initial_args),
        _span("optimizer.lifecycle.evaluator_return", 950.0, 0.001, args=initial_args),
        _span(
            "optimizer.accepted_iteration",
            1000.0,
            1000.0,
            args={"step_num": "1"},
        ),
        _span("optimizer.lifecycle.evaluator_entry", 1040.0, 0.001, args=trial_args),
        _span(PhaseId.HOST_H2D_SUBMIT.value, 1050.0, 50.0, args=trial_args),
        _span(
            "MemcpyH2D",
            1060.0,
            10.0,
            pid=_DEVICE_PID,
            args={
                "name": "MemcpyH2D",
                "memcpy_details": (
                    "kind_src:pinned kind_dst:device size:64 dest:0 async:1"
                ),
            },
        ),
        _span(
            "newton_kernel",
            1120.0,
            80.0,
            pid=_DEVICE_PID,
            args={
                "name": "jit(f)/newton.residual_jvp/sin",
                "kernel_details": "regs:16",
            },
        ),
        _span(
            "biotsavart_kernel",
            1140.0,
            20.0,
            pid=_DEVICE_PID,
            args={
                "name": ("jit(f)/newton.residual_jvp/biotsavart.forward/reduce"),
                "kernel_details": "regs:32",
            },
        ),
        _span(
            "adjoint_kernel",
            1250.0,
            100.0,
            pid=_DEVICE_PID,
            args={
                "name": "jit(f)/adjoint.lu_solve/triangular_solve",
                "kernel_details": "regs:64",
            },
        ),
        _span(
            "fused_unknown",
            1400.0,
            50.0,
            pid=_DEVICE_PID,
            args={"name": "jit(f)/loop_fusion", "kernel_details": "regs:24"},
        ),
        _span("optimizer.lifecycle.device_ready", 1490.0, 0.001, args=trial_args),
        _span(PhaseId.HOST_D2H_MATERIALIZE.value, 1500.0, 50.0, args=trial_args),
        _span(
            "MemcpyD2H",
            1510.0,
            10.0,
            pid=_DEVICE_PID,
            args={
                "correlation_id": "2",
                "name": "MemcpyD2H",
                "memcpy_details": (
                    "kind_src:device kind_dst:pinned size:64 dest:0 async:1"
                ),
            },
        ),
        _span("optimizer.lifecycle.evaluator_return", 1560.0, 0.001, args=trial_args),
        {},
    ]
    initial = _host_evaluation_events(
        timestamps_ns=(5_800_000, 5_900_000, 5_950_000),
        evaluation_id="initial",
        parameter_sha256="a" * 64,
        kind=EvaluationKind.INITIAL,
        iteration=None,
    )
    trial = _host_evaluation_events(
        timestamps_ns=(6_040_000, 6_490_000, 6_560_000),
        evaluation_id="trial-1",
        parameter_sha256="b" * 64,
        kind=EvaluationKind.TRIAL,
        iteration=1,
    )
    host_events = tuple(
        HostEventRecord(
            sequence=index,
            event=record.event,
            timestamp_ns=record.timestamp_ns,
            evaluation=record.evaluation,
            attributes=record.attributes,
        )
        for index, record in enumerate((*initial, *trial))
    )
    return {
        "displayTimeUnit": "ns",
        "metadata": {"highres-ticks": True},
        "traceEvents": events,
    }, host_events


def _evaluation_documents() -> tuple[dict[str, object], ...]:
    return (
        {
            "evaluation_id": "initial",
            "lifecycle": EvaluationKind.INITIAL.value,
            "iteration": None,
        },
        {
            "evaluation_id": "trial-1",
            "lifecycle": EvaluationKind.TRIAL.value,
            "iteration": 1,
            "inner_evidence": {"newton_iterations": 4},
            "adjoint_evidence": {
                "dense_materializations": 1,
                "lu_factorizations": 1,
                "lu_solves": 12,
                "refinement_corrections": 1,
                "adjoint_executions": 1,
            },
        },
    )


class TestTraceSummarizer:
    def test_separates_host_clock_and_unioned_device_intervals(self) -> None:
        document, host_events = _fixture()

        summary = summarize_trace_document(
            document, host_events, child_id="profile-1", expected_iterations=1
        )

        iteration = summary.iterations[0]
        assert iteration.host_control_gap_ns == 90_000
        assert iteration.cuda_memcpy_ns == 20_000
        assert iteration.host_boundary_ns == 110_000
        assert iteration.newton_adjoint_ns == 180_000
        assert iteration.unattributed_ns == 50_000
        assert iteration.device_active_ns == 250_000
        assert iteration.active_ns == 340_000
        assert iteration.device_overlap_ns == 20_000
        assert (
            dict(iteration.phase_active_ns)[PhaseId.BIOTSAVART_FORWARD.value] == 20_000
        )
        assert summary.clock_correlation_valid
        assert not summary.required_phase_families_present
        assert not summary.semantic_counts_available
        assert iteration.semantic_solver_counts == ()
        assert (
            dict(iteration.device_interval_counts)[PhaseId.ADJOINT_LU_SOLVE.value] == 1
        )
        assert PhaseId.ADJOINT_DENSE_MATRIX.value in iteration.missing_required_phases

    def test_exact_solver_counts_and_biot_kernel_groups_are_separate(self) -> None:
        document, host_events = _fixture()
        trace_events = document["traceEvents"]
        assert isinstance(trace_events, list)
        trace_events.insert(
            -1,
            _span(
                "biotsavart_vjp_kernel",
                1360.0,
                20.0,
                pid=_DEVICE_PID,
                args={
                    "name": "jit(f)/adjoint.implicit_coil_vjp/biotsavart.vjp/reduce",
                    "kernel_details": "regs:32",
                },
            ),
        )

        summary = summarize_trace_document(
            document,
            host_events,
            child_id="profile-1",
            expected_iterations=1,
            evaluation_documents=_evaluation_documents(),
        )

        iteration = summary.iterations[0]
        assert summary.semantic_counts_available
        assert dict(iteration.semantic_solver_counts) == {
            "adjoint_executions": 1,
            "dense_materializations": 1,
            "lu_factorizations": 1,
            "lu_solves": 12,
            "newton_iterations": 4,
            "refinement_corrections": 1,
        }
        assert dict(iteration.device_kernel_group_counts) == {
            PhaseId.BIOTSAVART_FORWARD.value: 1,
            PhaseId.BIOTSAVART_VJP.value: 1,
        }

    def test_missing_biot_group_keeps_exact_solver_counts_nonpromoting(self) -> None:
        document, host_events = _fixture()

        summary = summarize_trace_document(
            document,
            host_events,
            child_id="profile-1",
            expected_iterations=1,
            evaluation_documents=_evaluation_documents(),
        )

        iteration = summary.iterations[0]
        assert not summary.semantic_counts_available
        assert dict(iteration.semantic_solver_counts)["newton_iterations"] == 4
        assert dict(iteration.device_kernel_group_counts) == {
            PhaseId.BIOTSAVART_FORWARD.value: 1,
            PhaseId.BIOTSAVART_VJP.value: 0,
        }

    def test_rejects_nonbijective_evaluation_documents(self) -> None:
        document, host_events = _fixture()

        with pytest.raises(
            TraceSummaryError, match="semantic_count_correlation_invalid"
        ):
            summarize_trace_document(
                document,
                host_events,
                child_id="profile-1",
                expected_iterations=1,
                evaluation_documents=_evaluation_documents()[1:],
            )

    def test_memcpy_schema_does_not_invent_an_args_name(self) -> None:
        document, host_events = _fixture()
        trace_events = document["traceEvents"]
        assert isinstance(trace_events, list)
        for event in trace_events:
            if event.get("name") != "MemcpyH2D":
                continue
            args = event["args"]
            assert isinstance(args, dict)
            del args["name"]

        summary = summarize_trace_document(
            document, host_events, child_id="profile-1", expected_iterations=1
        )

        assert summary.iterations[0].cuda_memcpy_ns == 20_000

    def test_empty_complete_event_name_attributes_from_args(self) -> None:
        document, host_events = _fixture()
        trace_events = document["traceEvents"]
        assert isinstance(trace_events, list)
        newton_event = next(
            event for event in trace_events if event.get("name") == "newton_kernel"
        )
        newton_event["name"] = ""

        summary = summarize_trace_document(
            document, host_events, child_id="profile-1", expected_iterations=1
        )

        assert summary.iterations[0].newton_adjoint_ns == 180_000

    def test_kernel_without_args_name_attributes_from_hlo_op(self) -> None:
        document, host_events = _fixture()
        trace_events = document["traceEvents"]
        assert isinstance(trace_events, list)
        newton_event = next(
            event for event in trace_events if event.get("name") == "newton_kernel"
        )
        args = newton_event["args"]
        assert isinstance(args, dict)
        args["hlo_op"] = "jit(f)/newton.residual_jvp/sin"
        del args["name"]

        summary = summarize_trace_document(
            document, host_events, child_id="profile-1", expected_iterations=1
        )

        assert summary.iterations[0].newton_adjoint_ns == 180_000

    def test_unlabeled_kernel_inherits_unique_scope_range_owner(self) -> None:
        document, host_events = _fixture()
        trace_events = document["traceEvents"]
        assert isinstance(trace_events, list)
        trace_events.insert(
            -1,
            _span(
                PhaseId.NEWTON_LINEAR_SOLVE.value,
                1100.0,
                1.0,
                args={
                    "name": PhaseId.NEWTON_LINEAR_SOLVE.value,
                    "scope_range_id": "inherited-range",
                },
            ),
        )
        trace_events.insert(
            -1,
            _span(
                "range_owned_kernel",
                1370.0,
                20.0,
                pid=_DEVICE_PID,
                args={
                    "name": "jit(f)/loop_fusion",
                    "kernel_details": "regs:16",
                    "scope_range_id": "inherited-range",
                },
            ),
        )

        summary = summarize_trace_document(
            document, host_events, child_id="profile-1", expected_iterations=1
        )

        iteration = summary.iterations[0]
        assert (
            dict(iteration.device_interval_counts)[PhaseId.NEWTON_LINEAR_SOLVE.value]
            == 1
        )
        assert iteration.newton_adjoint_ns == 200_000
        assert iteration.unattributed_ns == 50_000

    def test_unlabeled_memcpy_inherits_unique_scope_range_owner(self) -> None:
        document, host_events = _fixture()
        trace_events = document["traceEvents"]
        assert isinstance(trace_events, list)
        trace_events.insert(
            -1,
            _span(
                PhaseId.ADJOINT_LU_SOLVE.value,
                1100.0,
                1.0,
                args={
                    "name": PhaseId.ADJOINT_LU_SOLVE.value,
                    "scope_range_id": "memcpy-range",
                },
            ),
        )
        trace_events.insert(
            -1,
            _span(
                "MemcpyD2D",
                1370.0,
                20.0,
                pid=_DEVICE_PID,
                args={
                    "correlation_id": "unmatched",
                    "memcpy_details": (
                        "kind_src:device kind_dst:device size:64 dest:0 async:1"
                    ),
                    "scope_range_id": "memcpy-range",
                },
            ),
        )

        summary = summarize_trace_document(
            document, host_events, child_id="profile-1", expected_iterations=1
        )

        iteration = summary.iterations[0]
        assert (
            dict(iteration.device_interval_counts)[PhaseId.ADJOINT_LU_SOLVE.value] == 2
        )
        assert iteration.newton_adjoint_ns == 200_000

    def test_direct_nested_scope_precedes_inherited_range_owner(self) -> None:
        document, host_events = _fixture()
        trace_events = document["traceEvents"]
        assert isinstance(trace_events, list)
        trace_events.insert(
            -1,
            _span(
                PhaseId.NEWTON_RESIDUAL_JVP.value,
                1100.0,
                1.0,
                args={
                    "name": PhaseId.NEWTON_RESIDUAL_JVP.value,
                    "scope_range_id": "nested-range",
                },
            ),
        )
        trace_events.insert(
            -1,
            _span(
                "direct_nested_kernel",
                1370.0,
                20.0,
                pid=_DEVICE_PID,
                args={
                    "name": "jit(f)/loop_fusion",
                    "hlo_op": ("jit(f)/newton.residual_jvp/biotsavart.forward/reduce"),
                    "kernel_details": "regs:16",
                    "scope_range_id": "nested-range",
                },
            ),
        )

        summary = summarize_trace_document(
            document, host_events, child_id="profile-1", expected_iterations=1
        )

        iteration = summary.iterations[0]
        assert (
            dict(iteration.device_interval_counts)[PhaseId.BIOTSAVART_FORWARD.value]
            == 2
        )
        assert (
            dict(iteration.device_interval_counts)[PhaseId.NEWTON_RESIDUAL_JVP.value]
            == 1
        )

    def test_rejects_conflicting_direct_labels_for_one_scope_range(self) -> None:
        document, host_events = _fixture()
        trace_events = document["traceEvents"]
        assert isinstance(trace_events, list)
        trace_events[-1:-1] = [
            _span(
                PhaseId.NEWTON_LINEAR_SOLVE.value,
                1100.0,
                1.0,
                args={
                    "name": PhaseId.NEWTON_LINEAR_SOLVE.value,
                    "scope_range_id": "conflicting-range",
                },
            ),
            _span(
                PhaseId.ADJOINT_LU_SOLVE.value,
                1102.0,
                1.0,
                args={
                    "name": PhaseId.ADJOINT_LU_SOLVE.value,
                    "scope_range_id": "conflicting-range",
                },
            ),
        ]

        with pytest.raises(TraceSummaryError, match="scope_range_attribution_invalid"):
            summarize_trace_document(
                document, host_events, child_id="profile-1", expected_iterations=1
            )

    def test_unlabeled_kernel_without_range_label_stays_unattributed(self) -> None:
        document, host_events = _fixture()
        trace_events = document["traceEvents"]
        assert isinstance(trace_events, list)
        unknown_event = next(
            event for event in trace_events if event.get("name") == "fused_unknown"
        )
        args = unknown_event["args"]
        assert isinstance(args, dict)
        args["scope_range_id"] = "unlabeled-range"

        summary = summarize_trace_document(
            document, host_events, child_id="profile-1", expected_iterations=1
        )

        assert summary.iterations[0].unattributed_ns == 50_000

    def test_scope_range_does_not_infer_owner_from_hlo_module(self) -> None:
        document, host_events = _fixture()
        trace_events = document["traceEvents"]
        assert isinstance(trace_events, list)
        trace_events.insert(
            -1,
            _span(
                "hlo_only_observation",
                1100.0,
                1.0,
                args={
                    "hlo_module": "jit_biotsavart_forward",
                    "scope_range_id": "hlo-only-range",
                },
            ),
        )
        trace_events.insert(
            -1,
            _span(
                "hlo_range_kernel",
                1370.0,
                20.0,
                pid=_DEVICE_PID,
                args={
                    "name": "jit(f)/loop_fusion",
                    "kernel_details": "regs:16",
                    "scope_range_id": "hlo-only-range",
                },
            ),
        )

        summary = summarize_trace_document(
            document, host_events, child_id="profile-1", expected_iterations=1
        )

        assert summary.iterations[0].unattributed_ns == 70_000

    def test_rejects_non_string_complete_event_name(self) -> None:
        document, host_events = _fixture()
        trace_events = document["traceEvents"]
        assert isinstance(trace_events, list)
        newton_event = next(
            event for event in trace_events if event.get("name") == "newton_kernel"
        )
        newton_event["name"] = 0

        with pytest.raises(
            TraceSummaryError, match=r"traceEvents\[[0-9]+\]\.name must be a string"
        ):
            summarize_trace_document(
                document, host_events, child_id="profile-1", expected_iterations=1
            )

    def test_memcpy_correlation_survives_async_device_interval_tail(self) -> None:
        document, host_events = _fixture()
        trace_events = document["traceEvents"]
        assert isinstance(trace_events, list)
        h2d = next(event for event in trace_events if event.get("name") == "MemcpyH2D")
        h2d["ts"] = 1105.0
        trace_events.insert(
            -1,
            _span(
                "MemcpyH2D",
                1060.0,
                2.0,
                args={
                    "context_id": "$$1",
                    "correlation_id": "1",
                    "memcpy_details": (
                        "kind_src:unknown kind_dst:unknown size:64 dest:0 async:1"
                    ),
                },
            ),
        )

        summary = summarize_trace_document(
            document, host_events, child_id="profile-1", expected_iterations=1
        )

        assert summary.iterations[0].cuda_memcpy_ns == 20_000

    def test_cross_category_overlap_becomes_unattributed(self) -> None:
        document, host_events = _fixture()
        trace_events = document["traceEvents"]
        assert isinstance(trace_events, list)
        trace_events.insert(
            -1,
            _span(
                "overlapping_other",
                1180.0,
                70.0,
                pid=_DEVICE_PID,
                args={
                    "name": "jit(f)/optimizer.lifecycle/other",
                    "kernel_details": "regs:16",
                },
            ),
        )

        summary = summarize_trace_document(
            document, host_events, child_id="profile-1", expected_iterations=1
        )

        iteration = summary.iterations[0]
        assert iteration.newton_adjoint_ns == 160_000
        assert iteration.other_attributed_ns == 50_000
        assert iteration.unattributed_ns == 70_000
        assert iteration.device_active_ns == 300_000

    def test_rejects_unknown_top_level_trace_schema(self) -> None:
        document, host_events = _fixture()
        document["unexpected"] = True

        with pytest.raises(TraceSummaryError, match="unknown_trace_schema"):
            summarize_trace_document(
                document, host_events, child_id="profile-1", expected_iterations=1
            )

    def test_rejects_unknown_timeline_phase_segment(self) -> None:
        document, host_events = _fixture()
        trace_events = document["traceEvents"]
        assert isinstance(trace_events, list)
        newton_event = next(
            event for event in trace_events if event.get("name") == "newton_kernel"
        )
        args = newton_event["args"]
        assert isinstance(args, dict)
        args["name"] = "jit(f)/newton.dense_jacobian/op"

        with pytest.raises(TraceSummaryError, match="unknown_phase_id"):
            summarize_trace_document(
                document, host_events, child_id="profile-1", expected_iterations=1
            )

    def test_all_scope_metadata_participates_in_direct_ambiguity_detection(
        self,
    ) -> None:
        document, host_events = _fixture()
        trace_events = document["traceEvents"]
        assert isinstance(trace_events, list)
        newton_event = next(
            event for event in trace_events if event.get("name") == "newton_kernel"
        )
        args = newton_event["args"]
        assert isinstance(args, dict)
        args["hlo_op"] = PhaseId.ADJOINT_LU_SOLVE.value

        summary = summarize_trace_document(
            document, host_events, child_id="profile-1", expected_iterations=1
        )

        iteration = summary.iterations[0]
        assert iteration.newton_adjoint_ns == 100_000
        assert iteration.unattributed_ns == 130_000

    def test_rejects_missing_jax_kernel_schema_field(self) -> None:
        document, host_events = _fixture()
        trace_events = document["traceEvents"]
        assert isinstance(trace_events, list)
        newton_event = next(
            event for event in trace_events if event.get("name") == "newton_kernel"
        )
        args = newton_event["args"]
        assert isinstance(args, dict)
        del args["context_id"]

        with pytest.raises(TraceSummaryError, match="unknown_trace_schema"):
            summarize_trace_document(
                document, host_events, child_id="profile-1", expected_iterations=1
            )

    def test_rejects_compilation_event_inside_accepted_iteration(self) -> None:
        document, host_events = _fixture()
        trace_events = document["traceEvents"]
        assert isinstance(trace_events, list)
        trace_events.insert(-1, _span("PJRT_Client_Compile", 970.0, 5.0))

        with pytest.raises(TraceSummaryError, match="compilation_in_measurement"):
            summarize_trace_document(
                document, host_events, child_id="profile-1", expected_iterations=1
            )

    def test_rejects_missing_lifecycle_point(self) -> None:
        document, host_events = _fixture()

        with pytest.raises(TraceSummaryError, match="host_correlation_invalid"):
            summarize_trace_document(
                document,
                host_events[:-1],
                child_id="profile-1",
                expected_iterations=1,
            )

    def test_rejects_unknown_lifecycle_event_name(self) -> None:
        document, host_events = _fixture()
        trace_events = document["traceEvents"]
        assert isinstance(trace_events, list)
        trace_events.insert(
            -1,
            _span(
                "optimizer.lifecycle.unowned",
                1570.0,
                0.001,
                args=_lifecycle_args("trial-1", "b" * 64, EvaluationKind.TRIAL, 1),
            ),
        )

        with pytest.raises(TraceSummaryError, match="host_correlation_invalid"):
            summarize_trace_document(
                document, host_events, child_id="profile-1", expected_iterations=1
            )

    def test_rejects_duplicate_evaluation_group(self) -> None:
        document, host_events = _fixture()
        duplicated = (
            *host_events,
            *(
                record
                for record in host_events
                if record.evaluation.evaluation_id == "trial-1"
            ),
        )

        with pytest.raises(TraceSummaryError, match="host_correlation_invalid"):
            summarize_trace_document(
                document, duplicated, child_id="profile-1", expected_iterations=1
            )

    def test_rejects_lifecycle_anchors_with_clock_drift(self) -> None:
        document, host_events = _fixture()
        changed = list(host_events)
        trial_entry = changed[3]
        changed[3] = HostEventRecord(
            sequence=trial_entry.sequence,
            event=trial_entry.event,
            timestamp_ns=trial_entry.timestamp_ns + 120_000,
            evaluation=trial_entry.evaluation,
            attributes=trial_entry.attributes,
        )

        with pytest.raises(TraceSummaryError, match="clock_correlation_invalid"):
            summarize_trace_document(
                document, changed, child_id="profile-1", expected_iterations=1
            )

    def test_rejects_reversed_host_gap(self) -> None:
        document, host_events = _fixture()
        changed = list(host_events)
        trial_entry = changed[3]
        changed[3] = HostEventRecord(
            sequence=trial_entry.sequence,
            event=trial_entry.event,
            timestamp_ns=10_100,
            evaluation=trial_entry.evaluation,
            attributes=trial_entry.attributes,
        )

        with pytest.raises(TraceSummaryError, match="host_correlation_invalid"):
            summarize_trace_document(
                document, changed, child_id="profile-1", expected_iterations=1
            )

    def test_rejects_device_interval_crossing_iteration_boundary(self) -> None:
        document, host_events = _fixture()
        trace_events = document["traceEvents"]
        assert isinstance(trace_events, list)
        trace_events.insert(
            -1,
            _span(
                "crossing",
                1990.0,
                20.0,
                pid=_DEVICE_PID,
                args={
                    "name": "jit(f)/newton.residual_jvp/crossing",
                    "kernel_details": "regs:16",
                },
            ),
        )

        with pytest.raises(TraceSummaryError, match="crosses"):
            summarize_trace_document(
                document, host_events, child_id="profile-1", expected_iterations=1
            )

    def test_summary_json_is_detached_from_internal_tuples(self) -> None:
        document, host_events = _fixture()
        summary = summarize_trace_document(
            document, host_events, child_id="profile-1", expected_iterations=1
        )

        first = summary.to_json()
        second = copy.deepcopy(first)
        iterations = first["iterations"]
        assert isinstance(iterations, list)
        first_iteration = iterations[0]
        assert isinstance(first_iteration, dict)
        first_iteration["active_ns"] = -1

        assert summary.to_json() == second


class TestTraceIntervalAttribution:
    def test_interval_union_does_not_sum_nested_or_overlapping_time(self) -> None:
        merged, duration_ns = union_intervals(
            (Interval(0, 10), Interval(2, 4), Interval(8, 15), Interval(20, 25))
        )

        assert merged == (Interval(0, 15), Interval(20, 25))
        assert duration_ns == 20

    def test_equally_deep_multi_owner_scope_is_ambiguous(self) -> None:
        attribution = unique_deepest_scope(
            (
                (PhaseId.ADJOINT_OUTER_VJP_RHS, PhaseId.BIOTSAVART_VJP),
                (PhaseId.NEWTON_RESIDUAL_JVP, PhaseId.BIOTSAVART_FORWARD),
            )
        )

        assert attribution.phase is None
        assert attribution.critical_phase is None
        assert attribution.ambiguous

    def test_deepest_leaf_retains_enclosing_critical_phase(self) -> None:
        attribution = unique_deepest_scope(
            ((PhaseId.ADJOINT_IMPLICIT_COIL_VJP, PhaseId.BIOTSAVART_VJP),)
        )

        assert attribution.phase is PhaseId.BIOTSAVART_VJP
        assert attribution.critical_phase is PhaseId.ADJOINT_IMPLICIT_COIL_VJP
        assert not attribution.ambiguous

    @staticmethod
    def _device_interval(
        start_ns: int, end_ns: int, category: str
    ) -> timeline_summary._DeviceInterval:
        phases = {
            "host_boundary": PhaseId.HOST_H2D_SUBMIT,
            "newton_adjoint": PhaseId.NEWTON_RESIDUAL_JVP,
            "other_attributed": PhaseId.OPTIMIZER_LIFECYCLE,
            "unattributed": None,
        }
        phase = phases[category]
        return timeline_summary._DeviceInterval(
            interval=Interval(start_ns, end_ns),
            kind="kernel",
            attribution=timeline_summary.ScopeAttribution(
                phase=phase,
                critical_phase=phase,
                ambiguous=False,
            ),
        )

    @staticmethod
    def _brute_force_exclusive_durations(
        intervals: tuple[timeline_summary._DeviceInterval, ...],
    ) -> tuple[dict[str, int], int]:
        durations = {
            "host_boundary": 0,
            "newton_adjoint": 0,
            "other_attributed": 0,
            "unattributed": 0,
        }
        endpoints = sorted(
            {
                endpoint
                for item in intervals
                for endpoint in (item.interval.start_ns, item.interval.end_ns)
            }
        )
        for start_ns, end_ns in zip(endpoints, endpoints[1:]):
            categories = {
                timeline_summary._critical_category(item)
                for item in intervals
                if item.interval.start_ns < end_ns and start_ns < item.interval.end_ns
            }
            if not categories:
                continue
            category = (
                next(iter(categories)) if len(categories) == 1 else "unattributed"
            )
            durations[category] += end_ns - start_ns
        return durations, sum(durations.values())

    def test_exclusive_category_sweep_respects_half_open_boundary_ties(self) -> None:
        intervals = (
            self._device_interval(0, 10, "host_boundary"),
            self._device_interval(2, 8, "host_boundary"),
            self._device_interval(10, 20, "newton_adjoint"),
            self._device_interval(15, 25, "other_attributed"),
            self._device_interval(25, 30, "unattributed"),
        )

        durations, active_ns = timeline_summary._exclusive_category_durations(intervals)

        assert durations == {
            "host_boundary": 10,
            "newton_adjoint": 5,
            "other_attributed": 5,
            "unattributed": 10,
        }
        assert active_ns == 30

    def test_exclusive_category_sweep_matches_randomized_brute_force(self) -> None:
        generator = random.Random(20260806)
        categories = (
            "host_boundary",
            "newton_adjoint",
            "other_attributed",
            "unattributed",
        )
        for _ in range(250):
            intervals = tuple(
                self._device_interval(
                    start_ns,
                    start_ns + generator.randint(1, 20),
                    generator.choice(categories),
                )
                for start_ns in (
                    generator.randint(0, 50) for _ in range(generator.randint(0, 30))
                )
            )

            assert timeline_summary._exclusive_category_durations(
                intervals
            ) == self._brute_force_exclusive_durations(intervals)
