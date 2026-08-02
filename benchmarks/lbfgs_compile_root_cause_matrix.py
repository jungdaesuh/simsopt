"""Run bounded L-BFGS compile/runtime root-cause probes.

Each probe is an independent child process.  The parent samples that exact
PID's RSS and terminates it on the declared wall/RSS limit, so a costly
objective compile cannot consume an unbounded amount of memory or time.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPILE_SHAPE = REPO_ROOT / "benchmarks" / "lbfgs_ondevice_compile_shape.py"
WatchdogVerdict = Literal["completed", "timeout", "rss_limit", "failed"]
JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True)
class MatrixCase:
    name: str
    objective: Literal["quadratic", "coil47"]
    dimension: int
    maxcor: int
    maxiter: int
    maxfun: int
    compile_only: bool
    kernel: str = "step_from_start_to_next_observable"
    skip_lowering: bool = False


def default_cases() -> tuple[MatrixCase, ...]:
    """Return the small matrix covering the declared diagnostic axes."""

    return (
        MatrixCase(
            "quadratic-maxcor10-short-warm",
            "quadratic",
            2,
            10,
            2,
            8,
            False,
            skip_lowering=True,
        ),
        MatrixCase(
            "quadratic-maxcor300-long-warm",
            "quadratic",
            2,
            300,
            20,
            80,
            False,
            skip_lowering=True,
        ),
        MatrixCase(
            "coil47-maxcor10-short-compile",
            "coil47",
            47,
            10,
            2,
            8,
            True,
        ),
        MatrixCase(
            "coil47-maxcor10-long-warm",
            "coil47",
            47,
            10,
            20,
            80,
            False,
            skip_lowering=True,
        ),
        MatrixCase(
            "coil47-maxcor300-short-compile",
            "coil47",
            47,
            300,
            2,
            8,
            True,
        ),
    )


def _current_rss_kib(pid: int) -> int | None:
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None


def _required_object(value: JsonValue | None, name: str) -> JsonObject:
    if not isinstance(value, dict):
        raise TypeError(f"compile-shape payload has no {name} object")
    return cast(JsonObject, value)


def _optional_object(value: JsonValue | None, name: str) -> JsonObject | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"compile-shape {name} is not an object")
    return cast(JsonObject, value)


def _required_list(value: JsonValue | None, name: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise TypeError(f"compile-shape payload has no {name} list")
    return cast(list[JsonValue], value)


def watchdog_verdict(
    *,
    returncode: int | None,
    timed_out: bool,
    rss_exceeded: bool,
) -> WatchdogVerdict:
    if timed_out:
        return "timeout"
    if rss_exceeded:
        return "rss_limit"
    if returncode == 0:
        return "completed"
    return "failed"


def _terminate_child(child: subprocess.Popen[bytes]) -> None:
    child.terminate()
    try:
        child.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5.0)


def payload_summary(payload: JsonObject) -> JsonObject:
    case = _required_object(payload.get("case"), "case")
    runtime_compile = _optional_object(
        payload.get("runtime_compile"), "runtime_compile"
    )
    repeated = _optional_object(
        payload.get("repeated_call_compile"), "repeated_call_compile"
    )
    summaries = _required_list(payload.get("summaries"), "summaries")
    lowered_text_bytes = 0
    lower_s = 0.0
    for row_value in summaries:
        if not isinstance(row_value, dict):
            continue
        row = cast(JsonObject, row_value)
        text_bytes = row.get("text_bytes")
        if isinstance(text_bytes, (int, float)):
            lowered_text_bytes += int(text_bytes)
        lower_value = row.get("lower_s")
        if isinstance(lower_value, (int, float)):
            lower_s += float(lower_value)
    result_summary = repeated.get("result_summary") if repeated else None
    objective_timing = _optional_object(
        payload.get("objective_timing"),
        "objective_timing",
    )
    iteration_progress = (
        result_summary.get("iteration_progress") if result_summary is not None else None
    )
    return {
        "objective_kind": case.get("objective_kind"),
        "compiled_executable_count": (
            None
            if runtime_compile is None
            else runtime_compile.get("compiled_executable_count")
        ),
        "compile_log_count": None
        if repeated is None
        else repeated.get("compile_log_count"),
        "recompiled_on_repeated_calls": (
            None if repeated is None else repeated.get("recompiled_on_repeated_calls")
        ),
        "peak_host_rss_bytes": (
            None
            if runtime_compile is None
            else runtime_compile.get("peak_host_rss_bytes")
        ),
        "result_summary": result_summary,
        "lowered_text_bytes": lowered_text_bytes,
        "lower_s": lower_s,
        "objective_timing": objective_timing,
        "iteration_progress": iteration_progress,
        "solver_run_seconds": (
            None if result_summary is None else result_summary.get("run_seconds")
        ),
    }


def _run_case(
    case: MatrixCase,
    *,
    output_json: Path,
    timeout_s: float,
    rss_limit_kib: int,
    poll_s: float,
) -> JsonObject:
    progress_json = _progress_path(output_json)
    command = [
        sys.executable,
        str(COMPILE_SHAPE),
        "--objective",
        case.objective,
        "--dimension",
        str(case.dimension),
        "--maxcor",
        str(case.maxcor),
        "--maxiter",
        str(case.maxiter),
        "--maxfun",
        str(case.maxfun),
        "--maxls",
        "20",
        "--output-json",
        str(output_json),
        "--progress-json",
        str(progress_json),
        "--kernel",
        case.kernel,
        "--measure-objective",
    ]
    if case.compile_only:
        command.append("--skip-runtime-compile")
    else:
        command.append("--run-solver")
    if case.skip_lowering:
        command.append("--skip-lowering-summary")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    started = time.perf_counter()
    samples: list[dict[str, float | int | None]] = []
    timed_out = False
    rss_exceeded = False
    while True:
        elapsed = time.perf_counter() - started
        rss_kib = _current_rss_kib(child.pid)
        samples.append({"elapsed_s": elapsed, "rss_kib": rss_kib})
        returncode = child.poll()
        if returncode is not None:
            break
        if rss_kib is not None and rss_kib > rss_limit_kib:
            rss_exceeded = True
            _terminate_child(child)
            break
        if elapsed >= timeout_s:
            timed_out = True
            _terminate_child(child)
            break
        time.sleep(poll_s)

    returncode = child.returncode
    elapsed = time.perf_counter() - started
    verdict = watchdog_verdict(
        returncode=returncode,
        timed_out=timed_out,
        rss_exceeded=rss_exceeded,
    )
    result: JsonObject = {
        "case": cast(JsonObject, asdict(case)),
        "command": cast(list[JsonValue], command),
        "pid": child.pid,
        "exit_code": returncode,
        "verdict": verdict,
        "elapsed_s": elapsed,
        "max_rss_kib": max(
            (
                int(sample["rss_kib"])
                for sample in samples
                if sample["rss_kib"] is not None
            ),
            default=None,
        ),
        "rss_samples": cast(list[JsonValue], samples),
        "output_json": str(output_json),
        "progress_json": str(progress_json),
    }
    if progress_json.exists():
        result["progress"] = cast(
            JsonValue,
            json.loads(progress_json.read_text(encoding="utf-8")),
        )
    if verdict == "completed":
        payload_value = cast(
            JsonValue, json.loads(output_json.read_text(encoding="utf-8"))
        )
        payload = _required_object(payload_value, "output")
        result["payload_summary"] = payload_summary(payload)
    return result


def _write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _progress_path(output_json: Path) -> Path:
    return output_json.with_name(f"{output_json.stem}.progress.json")


def main() -> int:
    cases = default_cases()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(".artifacts/lbfgs-ondevice/root-cause-matrix.json"),
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--rss-limit-kib", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--poll-s", type=float, default=0.2)
    parser.add_argument(
        "--case", choices=tuple(case.name for case in cases), action="append"
    )
    args = parser.parse_args()
    selected = (
        cases
        if args.case is None
        else tuple(case for case in cases if case.name in set(args.case))
    )
    results: list[JsonObject] = []
    for case in selected:
        case_output = args.output_json.with_name(
            f"{args.output_json.stem}.{case.name}.json"
        )
        results.append(
            _run_case(
                case,
                output_json=case_output,
                timeout_s=args.timeout_s,
                rss_limit_kib=args.rss_limit_kib,
                poll_s=args.poll_s,
            )
        )
    payload: JsonObject = {
        "schema_version": 1,
        "generated_at_epoch_s": time.time(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "source_git_head": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT, text=True
        ).strip(),
        "watchdog": cast(
            JsonObject,
            {
                "timeout_s": args.timeout_s,
                "rss_limit_kib": args.rss_limit_kib,
                "poll_s": args.poll_s,
                "termination": "TERM then KILL after 5 seconds",
            },
        ),
        "results": cast(list[JsonValue], results),
    }
    _write_json(args.output_json, payload)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "verdicts": [result["verdict"] for result in results],
            },
            sort_keys=True,
        )
    )
    return 0 if all(result["verdict"] == "completed" for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
