"""Native microbenchmark for the Biot-Savart field-cache bookkeeping seam.

This benchmark isolates the cache-access part of the old and new compute paths.
It does not measure the Biot-Savart kernel math itself. The goal is to put
numbers on the storage rewrite:

- legacy compute bookkeeping: string formatting + std::map lookups
- indexed compute bookkeeping: slot-indexed cache access
- compatibility API: canonical recognized keys vs legacy fallback keys
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = Path(__file__).with_suffix(".cpp")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncoils", type=positive_int, default=16)
    parser.add_argument("--npoints", type=positive_int, default=400)
    parser.add_argument("--derivatives", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--warmup", type=nonnegative_int, default=128)
    parser.add_argument("--iterations", type=positive_int, default=5000)
    parser.add_argument("--samples", type=positive_int, default=9)
    parser.add_argument(
        "--cxx",
        type=str,
        default="",
        help="Override the C++ compiler. Defaults to $CXX, then c++, g++, clang++.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to write the full JSON payload.",
    )
    parser.add_argument(
        "--build-dir",
        type=str,
        default="",
        help="Optional directory to place the native benchmark binary.",
    )
    return parser.parse_args(argv)


def find_cxx(requested: str) -> str:
    candidate = requested or os.environ.get("CXX", "")
    if candidate:
        compiler = shutil.which(candidate)
        if compiler is None:
            raise RuntimeError(f"Requested compiler not found: {candidate}")
        return compiler

    for candidate in (shutil.which("c++"), shutil.which("g++"), shutil.which("clang++")):
        if candidate is not None:
            return candidate
    raise RuntimeError("No C++ compiler found. Install c++, g++, or clang++.")


def build_compile_command(*, cxx: str, source: Path, output: Path) -> list[str]:
    include_dirs = (
        REPO_ROOT,
        REPO_ROOT / "src" / "simsoptpp",
        REPO_ROOT / "thirdparty" / "fmt" / "include",
        REPO_ROOT / "thirdparty" / "xtensor" / "include",
        REPO_ROOT / "thirdparty" / "xtl" / "include",
        REPO_ROOT / "thirdparty" / "xsimd" / "include",
    )
    command = [cxx, "-std=c++17", "-O3", "-DNDEBUG", "-w"]
    command.extend(f"-I{include_dir}" for include_dir in include_dirs)
    command.extend([str(source), "-o", str(output)])
    return command


def run_command(command: list[str], *, error_prefix: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{error_prefix}:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def compile_native_benchmark(*, cxx: str, build_dir: Path) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    output = build_dir / "field_cache_hot_path_benchmark"
    command = build_compile_command(cxx=cxx, source=SOURCE_PATH, output=output)
    run_command(command, error_prefix="Native benchmark compile failed")
    return output


def run_native_benchmark(binary: Path, args: argparse.Namespace) -> dict[str, object]:
    command = [
        str(binary),
        "--ncoils",
        str(args.ncoils),
        "--npoints",
        str(args.npoints),
        "--derivatives",
        str(args.derivatives),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--samples",
        str(args.samples),
    ]
    result = run_command(command, error_prefix="Native benchmark run failed")
    return json.loads(result.stdout)


def run_benchmark_with_build_dir(
    *,
    build_dir: Path,
    compiler: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    binary = compile_native_benchmark(cxx=compiler, build_dir=build_dir)
    return run_native_benchmark(binary, args)


def git_sha() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def speedup_ratio(numerator_us: float, denominator_us: float) -> float:
    return numerator_us / denominator_us


def build_payload(
    native_payload: dict[str, object],
    *,
    compiler: str,
) -> dict[str, object]:
    legacy_compute = native_payload["legacy_compute_bookkeeping"]
    indexed_compute = native_payload["indexed_compute_bookkeeping"]
    compat_canonical = native_payload["compat_canonical_get_or_create"]
    compat_unknown = native_payload["compat_unknown_get_or_create"]
    return {
        "title": "Biot-Savart field-cache hot-path benchmark",
        "repo_sha": git_sha(),
        "platform": platform.platform(),
        "compiler": compiler,
        "source": str(SOURCE_PATH.relative_to(REPO_ROOT)),
        "config": native_payload["config"],
        "legacy_compute_bookkeeping": legacy_compute,
        "indexed_compute_bookkeeping": indexed_compute,
        "compat_canonical_get_or_create": compat_canonical,
        "compat_unknown_get_or_create": compat_unknown,
        "speedups": {
            "legacy_vs_indexed_compute": speedup_ratio(
                legacy_compute["median_us"],
                indexed_compute["median_us"],
            ),
            "compat_unknown_vs_canonical": speedup_ratio(
                compat_unknown["median_us"],
                compat_canonical["median_us"],
            ),
        },
    }


def format_summary(payload: dict[str, object]) -> str:
    config = payload["config"]
    legacy_compute = payload["legacy_compute_bookkeeping"]
    indexed_compute = payload["indexed_compute_bookkeeping"]
    compat_canonical = payload["compat_canonical_get_or_create"]
    compat_unknown = payload["compat_unknown_get_or_create"]
    speedups = payload["speedups"]
    return "\n".join(
        [
            "Biot-Savart field-cache hot-path benchmark",
            f"repo_sha: {payload['repo_sha']}",
            f"compiler: {payload['compiler']}",
            (
                "config: "
                f"ncoils={config['ncoils']} "
                f"npoints={config['npoints']} "
                f"derivatives={config['derivatives']} "
                f"warmup={config['warmup']} "
                f"iterations={config['iterations']} "
                f"samples={config['samples']}"
            ),
            (
                "legacy compute bookkeeping: "
                f"{legacy_compute['median_us']:.3f} us median, "
                f"{legacy_compute['mean_us']:.3f} us mean"
            ),
            (
                "indexed compute bookkeeping: "
                f"{indexed_compute['median_us']:.3f} us median, "
                f"{indexed_compute['mean_us']:.3f} us mean"
            ),
            (
                "legacy/indexed speedup: "
                f"{speedups['legacy_vs_indexed_compute']:.2f}x"
            ),
            (
                "compat canonical get_or_create: "
                f"{compat_canonical['median_us']:.3f} us median"
            ),
            (
                "compat legacy fallback get_or_create: "
                f"{compat_unknown['median_us']:.3f} us median"
            ),
            (
                "compat fallback/canonical speedup: "
                f"{speedups['compat_unknown_vs_canonical']:.2f}x"
            ),
            "note: this isolates cache bookkeeping only, not Biot-Savart kernel math.",
        ]
    )


def main() -> None:
    args = parse_args()
    compiler = find_cxx(args.cxx)
    if args.build_dir:
        native_payload = run_benchmark_with_build_dir(
            build_dir=Path(args.build_dir).resolve(),
            compiler=compiler,
            args=args,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="field-cache-hot-path-") as tmpdir:
            native_payload = run_benchmark_with_build_dir(
                build_dir=Path(tmpdir),
                compiler=compiler,
                args=args,
            )
    payload = build_payload(native_payload, compiler=compiler)
    print(format_summary(payload))
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
