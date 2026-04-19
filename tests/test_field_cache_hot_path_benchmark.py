from pathlib import Path

import pytest

from benchmarks import field_cache_hot_path_benchmark as benchmark


def test_build_compile_command_includes_repo_headers(tmp_path):
    output = tmp_path / "field-cache-hot-path"
    command = benchmark.build_compile_command(
        cxx="/usr/bin/c++",
        source=benchmark.SOURCE_PATH,
        output=output,
    )

    assert command[:5] == ["/usr/bin/c++", "-std=c++17", "-O3", "-DNDEBUG", "-w"]
    assert f"-I{benchmark.REPO_ROOT}" in command
    assert f"-I{benchmark.REPO_ROOT / 'thirdparty' / 'fmt' / 'include'}" in command
    assert f"-I{benchmark.REPO_ROOT / 'thirdparty' / 'xtensor' / 'include'}" in command
    assert command[-3:] == [str(benchmark.SOURCE_PATH), "-o", str(output)]


def test_format_summary_reports_speedups():
    payload = {
        "repo_sha": "deadbeef",
        "compiler": "/usr/bin/c++",
        "config": {
            "ncoils": 16,
            "npoints": 400,
            "derivatives": 2,
            "warmup": 128,
            "iterations": 5000,
            "samples": 9,
        },
        "legacy_compute_bookkeeping": {"median_us": 30.0, "mean_us": 31.0},
        "indexed_compute_bookkeeping": {"median_us": 2.0, "mean_us": 2.1},
        "compat_canonical_get_or_create": {"median_us": 0.3},
        "compat_unknown_get_or_create": {"median_us": 1.2},
        "speedups": {
            "legacy_vs_indexed_compute": 15.0,
            "compat_unknown_vs_canonical": 4.0,
        },
    }

    summary = benchmark.format_summary(payload)

    assert "legacy/indexed speedup: 15.00x" in summary
    assert "compat fallback/canonical speedup: 4.00x" in summary
    assert "note: this isolates cache bookkeeping only" in summary


def test_source_path_tracks_cpp_peer_file():
    assert benchmark.SOURCE_PATH == Path(benchmark.__file__).with_suffix(".cpp")


def test_find_cxx_honors_env_var(monkeypatch):
    monkeypatch.setenv("CXX", "/usr/bin/env")

    assert benchmark.find_cxx("") == "/usr/bin/env"


def test_parse_args_allows_zero_warmup():
    args = benchmark.parse_args(["--warmup", "0"])

    assert args.warmup == 0


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--iterations", "0"], "positive integer"),
        (["--samples", "0"], "positive integer"),
        (["--derivatives", "3"], "invalid choice"),
    ],
)
def test_parse_args_rejects_invalid_runtime_contract(argv, message, capsys):
    with pytest.raises(SystemExit):
        benchmark.parse_args(argv)
    captured = capsys.readouterr()
    assert message in captured.err
