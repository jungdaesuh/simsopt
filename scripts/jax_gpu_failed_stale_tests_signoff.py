#!/usr/bin/env python3
"""Fail-closed CUDA rerun harness for the 2026-06-05 stale-test cleanup plan."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET


_DEFAULT_INTEGRATION_PATHS = Path("docs/jax_gpu_integration_test_paths_2026-06-05.txt")
_DEFAULT_BATCH_PATHS_DIR = Path("docs/jax_gpu_integration_batches_2026-06-05")
_DEFAULT_BASELINE_SELECTORS = Path("docs/jax_gpu_failed_selectors_2026-06-05.txt")
_DEFAULT_RESULTS_DIR = Path(".artifacts/jax_gpu_failed_stale_tests_signoff")
_DEFAULT_CHUNK_COUNT = 21
_DEFAULT_TIMEOUT_SECONDS = 7200
_TIMEOUT_RETURNCODE = 124
_FULL_PYTEST_TRANSFER_GUARD = "log"
_STRICT_TRANSFER_GUARD = "disallow"
_RAW_JAX_TRANSFER_GUARD_ENVS = (
    "JAX_TRANSFER_GUARD",
    "JAX_TRANSFER_GUARD_DEVICE_TO_DEVICE",
    "JAX_TRANSFER_GUARD_DEVICE_TO_HOST",
    "JAX_TRANSFER_GUARD_HOST_TO_DEVICE",
)
_TRANSFER_GUARD_SNAPSHOT_ENVS = (
    "SIMSOPT_JAX_TRANSFER_GUARD",
    *_RAW_JAX_TRANSFER_GUARD_ENVS,
)
_ALLOWED_UNTRACKED_PREFIXES = (".artifacts/",)
_STALE_FAILURE_PATTERNS = (
    "surface_spec",
    "to_spec",
    "_call_boozer_residual",
    "_call_boozer_residual_ds",
    "_call_boozer_dresidual_dc",
    "simsopt_jax_adapters.geo.boozer_surface",
    "simsopt_jax.geo.optimizers.optimizer",
    "simsopt.geo.curvexyzfourier",
    "simsopt_jax_adapters.field.biotsavart_backend",
)
_SIGABRT_RETURNCODES = frozenset((-6, 134))


@dataclass(frozen=True)
class FocusedSelector:
    label: str
    selector: str


_FOCUSED_ABORT_REPROS = (
    FocusedSelector(
        label="batch_012_runtime_bundle_batched_value_and_grad",
        selector=(
            "tests/integration/test_single_stage_jax_cpu_reference.py::"
            "TestTraceableObjective::"
            "test_runtime_bundle_batched_value_and_grad_matches_serial"
        ),
    ),
    FocusedSelector(
        label="batch_012_runtime_bundle_rebuild_after_solver_option_change",
        selector=(
            "tests/integration/test_single_stage_jax_cpu_reference.py::"
            "TestTraceableObjective::"
            "test_runtime_bundle_rebuilds_after_solver_option_change_post_compile"
        ),
    ),
    FocusedSelector(
        label="batch_012_composite_gradient_finite_and_nonzero",
        selector=(
            "tests/integration/test_single_stage_jax_cpu_reference.py::"
            "TestCompositeGradientPipeline::"
            "test_composite_gradient_finite_and_nonzero"
        ),
    ),
)
_FOCUSED_LANE_SELECTORS = (
    FocusedSelector(
        label="batch_012_strict_cpu_non_qs_ratio_dj_transfer_guard",
        selector=(
            "tests/integration/test_single_stage_jax_cpu_reference.py::"
            "TestNonQSRatioValue::test_dj_allows_strict_transfer_guard"
        ),
    ),
    FocusedSelector(
        label="batch_012_strict_gpu_public_wrapper_dj_transfer_guard",
        selector=(
            "tests/integration/test_single_stage_jax_cpu_reference.py::"
            "TestCompositeObjective::"
            "test_public_wrapper_dj_boundaries_allow_strict_transfer_guard_real_fixture"
        ),
    ),
    FocusedSelector(
        label="batch_012_branch_stable_ondevice_m5_values",
        selector=(
            "tests/integration/test_single_stage_jax_cpu_reference.py::"
            "TestRealFixtureOndeviceM5Parity::"
            "test_real_fixture_ondevice_branch_stable_wrapper_values_match"
        ),
    ),
    FocusedSelector(
        label="batch_012_short_single_stage_stationary_outer_opt",
        selector=(
            "tests/integration/test_single_stage_jax_cpu_reference.py::"
            "TestShortSingleStageOptRun::"
            "test_outer_opt_accepts_stationary_initial_objective"
        ),
    ),
    FocusedSelector(
        label="batch_012_iotas_resolve_fd",
        selector=(
            "tests/integration/test_single_stage_jax_cpu_reference.py::"
            "TestIotasJAXResolveFD::test_iotas_resolve_fd"
        ),
    ),
)


@dataclass(frozen=True)
class SelectorFailure:
    batch: str
    classname: str
    name: str
    kind: str
    first_line: str

    def key(self) -> str:
        return "\t".join(
            (self.batch, self.classname, self.name, self.kind, self.first_line)
        )

    def row(self) -> str:
        return self.key()


@dataclass(frozen=True)
class CommandRecord:
    label: str
    returncode: int
    log_path: Path
    junit_path: Path | None


def _utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _resolve_repo(requested_repo: Path) -> Path:
    completed = subprocess.run(
        ["git", "-C", str(requested_repo), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(f"not a git checkout: {requested_repo}")
    return Path(completed.stdout.strip()).resolve()


def _repo_head(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit("failed to resolve git HEAD")
    return completed.stdout.strip()


def _require_clean_worktree(repo: Path) -> None:
    completed = subprocess.run(
        ["git", "-C", str(repo), "status", "--short", "--untracked-files=all"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit("failed to read git status")

    tracked_status: list[str] = []
    untracked_status: list[str] = []
    for line in completed.stdout.splitlines():
        if line.startswith("?? "):
            path = line[3:]
            if not path.startswith(_ALLOWED_UNTRACKED_PREFIXES):
                untracked_status.append(path)
        else:
            tracked_status.append(line)

    if tracked_status:
        raise SystemExit(
            "tracked worktree must be clean for CUDA signoff:\n"
            f"{chr(10).join(tracked_status)}"
        )
    if untracked_status:
        raise SystemExit(
            "non-artifact untracked paths would invalidate CUDA signoff:\n"
            f"{chr(10).join(untracked_status)}"
        )


def _require_cuda_runtime(python_bin: Path) -> None:
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("nvidia-smi is required for CUDA signoff")
    if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        raise SystemExit(f"PYTHON_BIN is not executable: {python_bin}")


def _base_env(repo: Path, results_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    for name in _RAW_JAX_TRANSFER_GUARD_ENVS:
        env.pop(name, None)
    cache_dir = results_dir / "jax_compilation_cache"
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "BLIS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PYTHONPATH": os.pathsep.join((str(repo), str(repo / "src"))),
            "JAX_ENABLE_X64": "1",
            "JAX_PLATFORMS": "cuda,cpu",
            "JAX_COMPILATION_CACHE_DIR": str(cache_dir),
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
            "JAX_LOG_COMPILES": "1",
            "JAX_TRACEBACK_FILTERING": "off",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "SIMSOPT_BACKEND_MODE": "jax_gpu_parity",
            "SIMSOPT_BACKEND_STRICT": "1",
            "SIMSOPT_JAX_PLATFORM": "cuda",
            "SIMSOPT_JAX_BACKEND": "cuda",
            "SIMSOPT_JAX_GPU_PREALLOCATE": "false",
            "SIMSOPT_JAX_TRANSFER_GUARD": _FULL_PYTEST_TRANSFER_GUARD,
        }
    )
    xla_flags = env.get("XLA_FLAGS", "")
    required_flags = (
        "--xla_gpu_exclude_nondeterministic_ops=true",
        "--xla_gpu_enable_llvm_module_compilation_parallelism=false",
    )
    flag_parts = tuple(part for part in xla_flags.split() if part)
    merged_parts = [*flag_parts]
    for flag in required_flags:
        if flag not in merged_parts:
            merged_parts.append(flag)
    env["XLA_FLAGS"] = " ".join(merged_parts)
    return env


def _strict_transfer_guard_env(env: dict[str, str]) -> dict[str, str]:
    strict_env = dict(env)
    strict_env["SIMSOPT_JAX_TRANSFER_GUARD"] = _STRICT_TRANSFER_GUARD
    return strict_env


def _transfer_guard_env_snapshot(env: dict[str, str]) -> dict[str, str]:
    return {
        name: env[name]
        for name in _TRANSFER_GUARD_SNAPSHOT_ENVS
        if name in env and env[name] != ""
    }


def _write_transfer_guard_env_policy(
    path: Path,
    *,
    full_pytest_env: dict[str, str],
    strict_probe_env: dict[str, str],
) -> None:
    payload = {
        "full_pytest_env": _transfer_guard_env_snapshot(full_pytest_env),
        "strict_probe_env": _transfer_guard_env_snapshot(strict_probe_env),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_logged(
    *,
    label: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    junit_path: Path | None = None,
    timeout_seconds: int,
    dry_run: bool,
) -> CommandRecord:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("DRY RUN: " + " ".join(command) + "\n", encoding="utf-8")
        (log_path.with_suffix(log_path.suffix + ".rc")).write_text(
            "0\n", encoding="utf-8"
        )
        return CommandRecord(label, 0, log_path, junit_path)

    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            log_file.write(f"\nTIMEOUT: {label} exceeded {timeout_seconds} seconds\n")
            returncode = _TIMEOUT_RETURNCODE
    (log_path.with_suffix(log_path.suffix + ".rc")).write_text(
        f"{returncode}\n", encoding="utf-8"
    )
    return CommandRecord(label, returncode, log_path, junit_path)


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _prepare_command_outputs(log_path: Path, junit_path: Path | None) -> None:
    _remove_if_exists(log_path)
    _remove_if_exists(log_path.with_suffix(log_path.suffix + ".rc"))
    if junit_path is not None:
        _remove_if_exists(junit_path)


def _clear_integration_batch_outputs(batch_dir: Path) -> None:
    for pattern in (
        "batch_*.xml",
        "batch_*.log",
        "batch_*.log.rc",
        "batch_*_paths.txt",
        "batch_*_focused_repro_deselectors.txt",
        "batch_*_focused_selector_deselectors.txt",
    ):
        for path in batch_dir.glob(pattern):
            _remove_if_exists(path)


def _clear_focused_selector_outputs(repro_dir: Path) -> None:
    for pattern in ("batch_*.xml", "batch_*.log", "batch_*.log.rc"):
        for path in repro_dir.glob(pattern):
            _remove_if_exists(path)


def _clear_focused_selector_root_outputs(results_dir: Path) -> None:
    for filename in (
        "focused_abort_repro_missing_selectors.txt",
        "focused_lane_missing_selectors.txt",
        "focused_abort_repro_selectors_with_missing_paths.txt",
        "focused_lane_selectors_with_missing_paths.txt",
    ):
        _remove_if_exists(results_dir / filename)


def _read_test_paths(path_file: Path) -> list[str]:
    paths = []
    for line in path_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            paths.append(stripped)
    return paths


def _read_batch_paths(
    *,
    integration_paths: Path,
    batch_paths_dir: Path,
    chunk_count: int,
) -> list[tuple[str, list[str]]]:
    requested_paths = _read_test_paths(integration_paths)
    batch_path_files = sorted(batch_paths_dir.glob("batch_*_paths.txt"))
    if not batch_path_files:
        return [
            (f"batch_{index:03d}", chunk)
            for index, chunk in enumerate(
                _split_chunks(requested_paths, chunk_count),
                start=1,
            )
        ]

    batches = [
        (path_file.stem.removesuffix("_paths"), _read_test_paths(path_file))
        for path_file in batch_path_files
    ]
    flattened = [test_path for _batch, paths in batches for test_path in paths]
    if flattened != requested_paths:
        raise SystemExit(
            f"batch path files in {batch_paths_dir} do not flatten to "
            f"{integration_paths}"
        )
    return batches


def _split_chunks(paths: list[str], chunk_count: int) -> list[list[str]]:
    if not paths:
        return []
    bounded_chunk_count = max(1, min(chunk_count, len(paths)))
    chunk_size = math.ceil(len(paths) / bounded_chunk_count)
    return [
        paths[index : index + chunk_size] for index in range(0, len(paths), chunk_size)
    ]


def _write_path_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _is_sigabrt_returncode(returncode: int) -> bool:
    return returncode in _SIGABRT_RETURNCODES


def _selector_path(selector: str) -> Path:
    return Path(selector.split("::", maxsplit=1)[0])


def _focused_signoff_selectors() -> tuple[FocusedSelector, ...]:
    return (*_FOCUSED_ABORT_REPROS, *_FOCUSED_LANE_SELECTORS)


def _focused_selector_deselectors(paths: list[str]) -> list[str]:
    path_prefixes = tuple(f"{path}::" for path in paths)
    return [
        focused.selector
        for focused in _focused_signoff_selectors()
        if focused.selector.startswith(path_prefixes)
    ]


def _first_failure_line(message: str, body: str) -> str:
    for line in f"{message}\n{body}".splitlines():
        stripped = line.strip()
        if stripped:
            return " ".join(stripped.split())
    return ""


def _extract_selectors(xml_paths: list[Path]) -> list[SelectorFailure]:
    rows: dict[tuple[str, str, str], SelectorFailure] = {}
    for xml_path in sorted(xml_paths):
        tree = ET.parse(xml_path)
        for case in tree.iter("testcase"):
            bad_children = [
                child for child in list(case) if child.tag in {"failure", "error"}
            ]
            if not bad_children:
                continue
            first = bad_children[0]
            classname = case.attrib.get("classname", "")
            name = case.attrib.get("name", "")
            selector = SelectorFailure(
                batch=xml_path.stem,
                classname=classname,
                name=name,
                kind=first.tag,
                first_line=_first_failure_line(
                    first.attrib.get("message", ""),
                    first.text or "",
                ),
            )
            rows[(selector.batch, selector.classname, selector.name)] = selector
    return [rows[key] for key in sorted(rows)]


def _write_selectors(
    path: Path, source_glob: str, selectors: list[SelectorFailure]
) -> None:
    lines = [
        f"# Generated from {source_glob}",
        "# Columns: batch, classname, name, kind, first_failure_or_error_line",
        "batch\tclassname\tname\tkind\tfirst_failure_or_error_line",
        *(selector.row() for selector in selectors),
    ]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _selector_rows(path: Path) -> set[str]:
    rows: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith("batch\t"):
            continue
        rows.add(line)
    return rows


def _write_selector_comparison(
    *,
    baseline_path: Path,
    current_path: Path,
    json_path: Path,
    markdown_path: Path,
) -> tuple[int, int]:
    baseline_rows = _selector_rows(baseline_path)
    current_rows = _selector_rows(current_path)
    cleared = sorted(baseline_rows - current_rows)
    new = sorted(current_rows - baseline_rows)
    payload: dict[str, object] = {
        "baseline_path": str(baseline_path),
        "current_path": str(current_path),
        "baseline_count": len(baseline_rows),
        "current_count": len(current_rows),
        "cleared_count": len(cleared),
        "new_count": len(new),
        "cleared": cleared,
        "new": new,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(
        "\n".join(
            (
                "# JAX GPU Failed Selector Comparison",
                "",
                f"- Baseline selectors: `{len(baseline_rows)}`",
                f"- Current selectors: `{len(current_rows)}`",
                f"- Cleared selectors: `{len(cleared)}`",
                f"- New selectors: `{len(new)}`",
                "",
                "See the JSON artifact for selector-level rows.",
                "",
            )
        ),
        encoding="utf-8",
    )
    return len(current_rows), len(new)


def _write_stale_failure_hits(
    selectors: list[SelectorFailure],
    output_path: Path,
) -> int:
    hits = []
    for selector in selectors:
        haystack = selector.row()
        if any(pattern in haystack for pattern in _STALE_FAILURE_PATTERNS):
            hits.append(selector.row())
    _write_path_lines(output_path, hits)
    return len(hits)


def _device_probe_code() -> str:
    return """
import jax

print("jax", jax.__version__)
print("backend", jax.default_backend())
print("devices", [(device.platform, str(device)) for device in jax.devices()])
if str(jax.default_backend()).lower() not in {"cuda", "gpu"}:
    raise SystemExit("expected CUDA/GPU default backend")
""".strip()


def _transfer_guard_probe_code() -> str:
    return """
import jax
import jax.numpy as jnp

with jax.transfer_guard_host_to_device("allow"):
    probe_value = jnp.asarray([1.0])
with jax.transfer_guard("disallow"):
    blocked = False
    try:
        print(probe_value)
    except RuntimeError:
        blocked = True
if not blocked:
    raise SystemExit("expected transfer guard to block implicit transfer")
print("transfer_guard disallow blocked implicit transfer")
""".strip()


def _pytest_command(
    python_bin: Path,
    junit_path: Path,
    selectors: list[str],
    *,
    verbose: bool,
    deselectors: list[str] | None = None,
) -> list[str]:
    command = [
        str(python_bin),
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        f"--junitxml={junit_path}",
        "--tb=short",
        "--disable-warnings",
    ]
    if verbose:
        command.extend(("-vv", "-s"))
    else:
        command.append("-q")
    if deselectors:
        command.extend(f"--deselect={selector}" for selector in deselectors)
    command.extend(selectors)
    return command


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CUDA proof packet for "
            "docs/jax_gpu_failed_stale_tests_impl_plan_2026-06-05.md."
        )
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path(".conda/jax/bin/python"),
    )
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--integration-paths",
        type=Path,
        default=_DEFAULT_INTEGRATION_PATHS,
    )
    parser.add_argument(
        "--batch-paths-dir",
        type=Path,
        default=_DEFAULT_BATCH_PATHS_DIR,
    )
    parser.add_argument(
        "--baseline-selectors",
        type=Path,
        default=_DEFAULT_BASELINE_SELECTORS,
    )
    parser.add_argument("--chunk-count", type=int, default=_DEFAULT_CHUNK_COUNT)
    parser.add_argument("--timeout-seconds", type=int, default=_DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--missing-path-policy",
        choices=("fail", "record"),
        default="fail",
    )
    parser.add_argument("--skip-clean-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    repo = _resolve_repo(args.repo)
    python_bin = args.python_bin
    if not python_bin.is_absolute():
        python_bin = repo / python_bin
    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = repo / results_dir
    integration_paths = args.integration_paths
    if not integration_paths.is_absolute():
        integration_paths = repo / integration_paths
    batch_paths_dir = args.batch_paths_dir
    if not batch_paths_dir.is_absolute():
        batch_paths_dir = repo / batch_paths_dir
    baseline_selectors = args.baseline_selectors
    if not baseline_selectors.is_absolute():
        baseline_selectors = repo / baseline_selectors

    results_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_clean_check and not args.dry_run:
        _require_clean_worktree(repo)
    if not args.dry_run:
        _require_cuda_runtime(python_bin)

    head = _repo_head(repo)
    (results_dir / "repo-head.txt").write_text(f"{head}\n", encoding="utf-8")
    (results_dir / "started-at.txt").write_text(f"{_utc_stamp()}\n", encoding="utf-8")
    env = _base_env(repo, results_dir)
    strict_transfer_env = _strict_transfer_guard_env(env)
    _write_transfer_guard_env_policy(
        results_dir / "transfer_guard_env_policy.json",
        full_pytest_env=env,
        strict_probe_env=strict_transfer_env,
    )
    records: list[CommandRecord] = []
    failures: list[str] = []

    if not args.dry_run:
        nvidia_log = results_dir / "nvidia-smi.txt"
        with nvidia_log.open("w", encoding="utf-8") as log_file:
            subprocess.run(["nvidia-smi"], stdout=log_file, stderr=subprocess.STDOUT)

    for label, code, command_env in (
        ("jax-device-probe", _device_probe_code(), env),
        ("transfer-guard-probe", _transfer_guard_probe_code(), strict_transfer_env),
    ):
        log_path = results_dir / f"{label}.log"
        _prepare_command_outputs(log_path, None)
        record = _run_logged(
            label=label,
            command=[str(python_bin), "-c", code],
            cwd=repo,
            env=command_env,
            log_path=log_path,
            timeout_seconds=args.timeout_seconds,
            dry_run=args.dry_run,
        )
        records.append(record)
        if record.returncode != 0:
            raise SystemExit(f"{label} failed; see {record.log_path}")

    pure_junit = results_dir / "pure_jax_tests.xml"
    pure_log = results_dir / "pure_jax_tests.log"
    _prepare_command_outputs(pure_log, pure_junit)
    records.append(
        _run_logged(
            label="pure-tests-jax",
            command=_pytest_command(
                python_bin, pure_junit, ["tests/jax"], verbose=False
            ),
            cwd=repo,
            env=env,
            log_path=pure_log,
            junit_path=pure_junit,
            timeout_seconds=args.timeout_seconds,
            dry_run=args.dry_run,
        )
    )

    focused_dir = results_dir / "focused_selectors"
    _clear_focused_selector_outputs(focused_dir)
    _clear_focused_selector_root_outputs(results_dir)
    focused_repro_selectors_with_missing_paths: list[str] = []
    focused_lane_selectors_with_missing_paths: list[str] = []
    for repro in _FOCUSED_ABORT_REPROS:
        if not (repo / _selector_path(repro.selector)).exists():
            focused_repro_selectors_with_missing_paths.append(repro.selector)
            continue
        junit_path = focused_dir / f"{repro.label}.xml"
        log_path = focused_dir / f"{repro.label}.log"
        _prepare_command_outputs(log_path, junit_path)
        records.append(
            _run_logged(
                label=f"{repro.label}-focused-repro",
                command=_pytest_command(
                    python_bin,
                    junit_path,
                    [repro.selector],
                    verbose=True,
                ),
                cwd=repo,
                env=env,
                log_path=log_path,
                junit_path=junit_path,
                timeout_seconds=args.timeout_seconds,
                dry_run=args.dry_run,
            )
        )
    for focused in _FOCUSED_LANE_SELECTORS:
        if not (repo / _selector_path(focused.selector)).exists():
            focused_lane_selectors_with_missing_paths.append(focused.selector)
            continue
        junit_path = focused_dir / f"{focused.label}.xml"
        log_path = focused_dir / f"{focused.label}.log"
        _prepare_command_outputs(log_path, junit_path)
        records.append(
            _run_logged(
                label=f"{focused.label}-focused-selector",
                command=_pytest_command(
                    python_bin,
                    junit_path,
                    [focused.selector],
                    verbose=True,
                ),
                cwd=repo,
                env=env,
                log_path=log_path,
                junit_path=junit_path,
                timeout_seconds=args.timeout_seconds,
                dry_run=args.dry_run,
            )
        )
    _write_path_lines(
        results_dir / "focused_abort_repro_selectors_with_missing_paths.txt",
        focused_repro_selectors_with_missing_paths,
    )
    _write_path_lines(
        results_dir / "focused_lane_selectors_with_missing_paths.txt",
        focused_lane_selectors_with_missing_paths,
    )
    if focused_repro_selectors_with_missing_paths and args.missing_path_policy == "fail":
        failures.append(
            f"{len(focused_repro_selectors_with_missing_paths)} focused abort "
            "repro selector paths are missing; rerun on the integration branch or use "
            "--missing-path-policy=record"
        )
    if focused_lane_selectors_with_missing_paths and args.missing_path_policy == "fail":
        failures.append(
            f"{len(focused_lane_selectors_with_missing_paths)} focused lane "
            "selector paths are missing; rerun on the integration branch or use "
            "--missing-path-policy=record"
        )

    requested_batches = _read_batch_paths(
        integration_paths=integration_paths,
        batch_paths_dir=batch_paths_dir,
        chunk_count=args.chunk_count,
    )
    requested_paths = [
        test_path for _batch, paths in requested_batches for test_path in paths
    ]
    present_paths = [path for path in requested_paths if (repo / path).exists()]
    missing_paths = [path for path in requested_paths if not (repo / path).exists()]
    _write_path_lines(results_dir / "integration_present_paths.txt", present_paths)
    _write_path_lines(results_dir / "integration_missing_paths.txt", missing_paths)
    skipped_batches: list[str] = []
    if missing_paths and args.missing_path_policy == "fail":
        failures.append(
            f"{len(missing_paths)} integration inventory paths are missing; "
            "rerun on the integration branch or use --missing-path-policy=record"
        )

    batch_dir = results_dir / "integration_batches"
    _clear_integration_batch_outputs(batch_dir)
    focused_repro_selectors = {repro.selector for repro in _FOCUSED_ABORT_REPROS}
    focused_lane_selectors = {focused.selector for focused in _FOCUSED_LANE_SELECTORS}
    focused_deselected_count = 0
    focused_repro_deselected_count = 0
    focused_lane_deselected_count = 0
    for batch_name, requested_chunk in requested_batches:
        chunk = [path for path in requested_chunk if (repo / path).exists()]
        if not chunk:
            skipped_batches.append(batch_name)
            continue
        deselectors = _focused_selector_deselectors(chunk)
        focused_deselected_count += len(deselectors)
        focused_repro_deselected_count += sum(
            selector in focused_repro_selectors for selector in deselectors
        )
        focused_lane_deselected_count += sum(
            selector in focused_lane_selectors for selector in deselectors
        )
        path_file = batch_dir / f"{batch_name}_paths.txt"
        _write_path_lines(path_file, chunk)
        if deselectors:
            _write_path_lines(
                batch_dir / f"{batch_name}_focused_selector_deselectors.txt",
                deselectors,
            )
        junit_path = batch_dir / f"{batch_name}.xml"
        log_path = batch_dir / f"{batch_name}.log"
        _prepare_command_outputs(log_path, junit_path)
        records.append(
            _run_logged(
                label=f"integration-{batch_name}",
                command=_pytest_command(
                    python_bin,
                    junit_path,
                    chunk,
                    verbose=False,
                    deselectors=deselectors,
                ),
                cwd=repo,
                env=env,
                log_path=log_path,
                junit_path=junit_path,
                timeout_seconds=args.timeout_seconds,
                dry_run=args.dry_run,
            )
        )
    _write_path_lines(results_dir / "integration_skipped_batches.txt", skipped_batches)

    integration_xmls = sorted(batch_dir.glob("batch_*.xml"))
    current_selectors: list[SelectorFailure] = []
    if integration_xmls:
        current_selectors = _extract_selectors(integration_xmls)
    current_selector_path = results_dir / "current_failed_selectors.tsv"
    _write_selectors(
        current_selector_path,
        str(batch_dir / "batch_*.xml"),
        current_selectors,
    )
    current_count, new_count = _write_selector_comparison(
        baseline_path=baseline_selectors,
        current_path=current_selector_path,
        json_path=results_dir / "failed_selector_comparison.json",
        markdown_path=results_dir / "failed_selector_comparison.md",
    )
    stale_hit_count = _write_stale_failure_hits(
        current_selectors,
        results_dir / "stale_failure_hits.tsv",
    )

    for record in records:
        if record.returncode != 0:
            failures.append(f"{record.label} returned {record.returncode}")
        if _is_sigabrt_returncode(record.returncode):
            failures.append(
                f"{record.label} hard-aborted with SIGABRT return code "
                f"{record.returncode}"
            )
    if current_count:
        failures.append(f"{current_count} integration failed/error selectors remain")
    if new_count:
        failures.append(f"{new_count} new selectors are not in the June 5 baseline")
    if stale_hit_count:
        failures.append(f"{stale_hit_count} selectors still match stale-owner patterns")

    summary: dict[str, object] = {
        "repo": str(repo),
        "repo_head": head,
        "results_dir": str(results_dir),
        "dry_run": bool(args.dry_run),
        "requested_integration_paths": len(requested_paths),
        "present_integration_paths": len(present_paths),
        "missing_integration_paths": len(missing_paths),
        "integration_batch_count": len(requested_batches),
        "skipped_integration_batch_count": len(skipped_batches),
        "focused_abort_repro_selectors": [
            repro.selector for repro in _FOCUSED_ABORT_REPROS
        ],
        "focused_lane_selectors": [
            focused.selector for focused in _FOCUSED_LANE_SELECTORS
        ],
        "focused_abort_repro_selectors_with_missing_paths": (
            focused_repro_selectors_with_missing_paths
        ),
        "focused_lane_selectors_with_missing_paths": (
            focused_lane_selectors_with_missing_paths
        ),
        "integration_focused_selector_deselect_count": focused_deselected_count,
        "integration_focused_repro_deselect_count": focused_repro_deselected_count,
        "integration_focused_lane_deselect_count": focused_lane_deselected_count,
        "batch_paths_dir": str(batch_paths_dir),
        "current_failed_selector_count": current_count,
        "new_failed_selector_count": new_count,
        "stale_failure_hit_count": stale_hit_count,
        "records": [
            {
                "label": record.label,
                "returncode": record.returncode,
                "log_path": str(record.log_path),
                "junit_path": None
                if record.junit_path is None
                else str(record.junit_path),
            }
            for record in records
        ],
        "failures": failures,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures:
        print(f"CUDA stale-test signoff failed: {results_dir}", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"CUDA stale-test signoff passed: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
