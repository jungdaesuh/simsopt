"""Unit coverage for the shared probe conventions.

``benchmarks/probe_conventions.py`` is bookkeeping, so all of it is testable
here on the ordinary test environment: the identity fields against this very
repository, build and host, the interleave against its own convention, the
environment pin against a poisoned base, the ledger against its append-only
rule, the artifact writer against its write-once and canonical-JSON rules, and
the cache lever against both of its branches.  Two things are exercised through
stand-ins rather than the real thing.  JAX state is read from a stub placed in
``sys.modules``: the module's contract is that it reads JAX state and never
creates it, and a test that imported JAX would both hide a regression in that
contract and put a CUDA context on a box that may be running someone else's
timed leg.  The git identity's ``-uall`` behaviour is exercised against a
scratch repository built in ``tmp_path``, because the live repository's
untracked set is whatever the working tree happens to hold and a test cannot
create an untracked directory inside it without dirtying the tree it is
measuring.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks import probe_conventions
from benchmarks.probe_conventions import (
    COMPILATION_CACHE_VARIABLE,
    CPU_BUSY_SAMPLE_SECONDS,
    OMP_SWEEP,
    PROBE_GRADE,
    REPO_ROOT,
    SCRUBBED_ENVIRONMENT_PREFIXES,
    THREAD_COUNT_VARIABLES,
    ProbeConventionError,
    append_leg_ledger,
    compile_cache_env,
    cpu_utilization_delta,
    gpu_compute_processes,
    interleave_schedule,
    observed_openmp_threads,
    pinned_environment,
    runtime_identity,
    write_probe_artifact,
)

_IDENTITY_FIELDS = (
    "lane",
    "hostname",
    "platform",
    "own_pid",
    "python_executable",
    "python_version",
    "git",
    "simsoptpp",
    "threading",
    "xla_flags",
    "timestamp_ns",
    "wallclock_utc",
    "loadavg",
    "jax_imported",
    "gpu",
    "gpu_compute_processes",
)
_JAX_FIELDS = ("jax_version", "jax_default_backend", "jax_enable_x64", "jax_devices")
_OBSERVED_LIBRARIES = ("numpy", "scipy", "jaxlib")


def _identity_without_jax(monkeypatch: pytest.MonkeyPatch, lane: str) -> dict:
    monkeypatch.delitem(sys.modules, "jax", raising=False)
    return runtime_identity(lane)


@pytest.fixture
def fixture_repository(tmp_path: Path) -> Path:
    """A scratch git repository with exactly one tracked file and one commit."""
    root = tmp_path / "fixture-repo"
    root.mkdir()

    def git(*arguments: str) -> None:
        subprocess.run(
            [
                "git",
                "-c",
                "user.email=probe@example.invalid",
                "-c",
                "user.name=Probe Conventions",
                "-c",
                "commit.gpgsign=false",
                *arguments,
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )

    git("init", "-q")
    (root / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-q", "-m", "seed")
    return root


def _stub_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch, *, devices: str = "", compute_apps: str = ""
) -> None:
    """Answer the two ``nvidia-smi`` queries from canned rows.

    Every other command still runs for real, so an identity captured under this
    stub keeps its genuine git block.  The two queries are answered separately
    because a single canned stdout would let a parser bug in one of them pass
    on the other's rows.
    """
    real_run = subprocess.run
    real_which = shutil.which

    def which(name: str) -> str | None:
        if name == "nvidia-smi":
            return "/usr/bin/nvidia-smi"
        return real_which(name)

    def run(command: list[str], **keywords: object) -> subprocess.CompletedProcess:
        if not command or not str(command[0]).endswith("nvidia-smi"):
            return real_run(command, **keywords)
        queried = next(item for item in command if item.startswith("--query-"))
        stdout = devices if queried.startswith("--query-gpu") else compute_apps
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(probe_conventions.shutil, "which", which)
    monkeypatch.setattr(probe_conventions.subprocess, "run", run)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_constants_are_the_declared_conventions() -> None:
    assert PROBE_GRADE == "diagnostic-not-certifying"
    assert OMP_SWEEP == (2, 4, 8, 16, 32, 48)
    assert COMPILATION_CACHE_VARIABLE == "JAX_COMPILATION_CACHE_DIR"
    assert CPU_BUSY_SAMPLE_SECONDS == 2.0
    assert SCRUBBED_ENVIRONMENT_PREFIXES == (
        "OMP_",
        "GOMP_",
        "KMP_",
        "MKL_",
        "OPENBLAS_",
        "NUMEXPR_",
        "VECLIB_",
        "BLIS_",
        "NUMBA_",
        "JAX_",
        "XLA_",
        "SIMSOPT_",
        "CUDA_",
        "MPI4PY_",
    )
    assert THREAD_COUNT_VARIABLES == (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )


def test_public_scrub_supersedes_the_private_threading_family() -> None:
    # The private name stays importable for the not-yet-migrated sibling, and
    # the public tuple is that sibling's own concatenation already assembled.
    assert (
        probe_conventions._NUMERICAL_ENVIRONMENT_PREFIXES
        == (SCRUBBED_ENVIRONMENT_PREFIXES[:9])
    )
    assert set(probe_conventions._NUMERICAL_ENVIRONMENT_PREFIXES) < set(
        SCRUBBED_ENVIRONMENT_PREFIXES
    )
    assert len(set(SCRUBBED_ENVIRONMENT_PREFIXES)) == len(SCRUBBED_ENVIRONMENT_PREFIXES)


# ---------------------------------------------------------------------------
# runtime_identity
# ---------------------------------------------------------------------------


def test_identity_carries_every_declared_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_without_jax(monkeypatch, "native")
    for field in _IDENTITY_FIELDS:
        assert field in identity, field
    assert identity["lane"] == "native"
    assert identity["hostname"]
    assert identity["own_pid"] == os.getpid()
    assert identity["python_executable"] == sys.executable
    assert isinstance(identity["timestamp_ns"], int)
    assert len(identity["loadavg"]) == 3
    assert all(isinstance(entry, float) for entry in identity["loadavg"])


def test_identity_wallclock_is_tz_aware_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = _identity_without_jax(monkeypatch, "native")
    stamped = datetime.fromisoformat(str(identity["wallclock_utc"]))
    assert stamped.tzinfo is not None
    assert stamped.utcoffset().total_seconds() == 0.0


def test_identity_git_block_binds_this_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_without_jax(monkeypatch, "native")
    git = identity["git"]
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert git["commit"] == expected_commit
    assert len(git["commit"]) == 40
    assert all(isinstance(entry, str) for entry in git["status"])
    # Every dirtied path is digested by content, and the digest is the file's
    # own -- this is the field that lets a receipt say which tree produced it.
    for path_text, digest in git["changed_file_sha256"].items():
        candidate = REPO_ROOT / path_text
        assert candidate.is_file()
        assert digest == hashlib.sha256(candidate.read_bytes()).hexdigest()
        assert any(path_text in entry for entry in git["status"])


def test_identity_git_block_enumerates_untracked_directories_file_by_file(
    monkeypatch: pytest.MonkeyPatch, fixture_repository: Path
) -> None:
    nested = fixture_repository / "untracked_dir" / "deeper"
    nested.mkdir(parents=True)
    (fixture_repository / "untracked_dir" / "one.txt").write_text("one\n")
    (nested / "two.txt").write_text("two\n")
    monkeypatch.setattr(probe_conventions, "REPO_ROOT", fixture_repository)
    git = _identity_without_jax(monkeypatch, "native")["git"]
    reported = {entry[3:] for entry in git["status"]}
    # ``-u`` alone would report the single entry ``untracked_dir/``, which is
    # not a path anything can hash; ``-uall`` names both files.
    assert "untracked_dir/" not in reported
    assert reported == {"untracked_dir/one.txt", "untracked_dir/deeper/two.txt"}
    assert set(git["changed_file_sha256"]) == reported
    assert git["changed_file_sha256"]["untracked_dir/deeper/two.txt"] == (
        hashlib.sha256(b"two\n").hexdigest()
    )


def test_identity_git_block_records_deletions_without_digesting_them(
    monkeypatch: pytest.MonkeyPatch, fixture_repository: Path
) -> None:
    (fixture_repository / "tracked.txt").unlink()
    monkeypatch.setattr(probe_conventions, "REPO_ROOT", fixture_repository)
    git = _identity_without_jax(monkeypatch, "native")["git"]
    assert git["status"] == [" D tracked.txt"]
    assert git["changed_file_sha256"] == {}


def test_identity_simsoptpp_block_digests_without_importing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "simsoptpp", raising=False)
    identity = _identity_without_jax(monkeypatch, "native")
    extension = identity["simsoptpp"]
    spec = importlib.util.find_spec("simsoptpp")
    if spec is None or spec.origin is None:
        assert extension is None
        return
    origin = Path(spec.origin)
    assert extension == {
        "path": str(origin),
        "sha256": hashlib.sha256(origin.read_bytes()).hexdigest(),
    }
    # The digest was taken off disk; the extension itself never got imported.
    assert "simsoptpp" not in sys.modules


def test_identity_library_versions_track_sys_modules_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_without_jax(monkeypatch, "native")
    for name in _OBSERVED_LIBRARIES:
        field = f"{name}_version"
        assert (field in identity) is (name in sys.modules), field
        if field in identity:
            assert isinstance(identity[field], str)


def test_identity_reports_a_library_version_once_the_process_has_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "jaxlib", SimpleNamespace(__version__="0.10.2"))
    identity = _identity_without_jax(monkeypatch, "native")
    assert identity["jaxlib_version"] == "0.10.2"


def test_identity_echoes_xla_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_exclude_nondeterministic_ops=true")
    assert _identity_without_jax(monkeypatch, "jax")["xla_flags"] == (
        "--xla_gpu_exclude_nondeterministic_ops=true"
    )
    monkeypatch.delenv("XLA_FLAGS")
    assert _identity_without_jax(monkeypatch, "jax")["xla_flags"] is None


def test_identity_records_observed_threading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.setenv("KMP_AFFINITY", "granularity=fine,compact")
    monkeypatch.setenv("PATH_NOT_NUMERICAL", "ignored")
    identity = _identity_without_jax(monkeypatch, "native")
    threading = identity["threading"]
    assert threading["environment"]["OMP_NUM_THREADS"] == "7"
    assert threading["environment"]["KMP_AFFINITY"] == "granularity=fine,compact"
    assert "PATH_NOT_NUMERICAL" not in threading["environment"]
    assert threading["cpu_affinity"] == sorted(threading["cpu_affinity"])
    assert threading["cpu_affinity"]
    assert threading["cpu_count"] >= 1


def test_identity_omits_jax_fields_when_jax_is_unimported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_without_jax(monkeypatch, "native")
    assert identity["jax_imported"] is False
    for field in _JAX_FIELDS:
        assert field not in identity


def test_identity_reads_jax_state_from_sys_modules_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = SimpleNamespace(
        __version__="0.10.2",
        default_backend=lambda: "cuda",
        config=SimpleNamespace(read=lambda name: name == "jax_enable_x64"),
        local_devices=lambda: [
            SimpleNamespace(platform="cuda", device_kind="NVIDIA GeForce RTX 5090")
        ],
    )
    monkeypatch.setitem(sys.modules, "jax", stub)
    identity = runtime_identity("jax")
    assert identity["jax_imported"] is True
    assert identity["jax_version"] == "0.10.2"
    assert identity["jax_default_backend"] == "cuda"
    assert identity["jax_enable_x64"] is True
    assert identity["jax_devices"] == [
        {"platform": "cuda", "kind": "NVIDIA GeForce RTX 5090"}
    ]


def test_identity_gpu_block_tracks_nvidia_smi_presence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_without_jax(monkeypatch, "native")
    devices = identity["gpu"]
    if shutil.which("nvidia-smi") is None:
        assert devices is None
        return
    assert isinstance(devices, list)
    for device in devices:
        assert set(device) == {"name", "uuid", "driver_version", "memory_total_mib"}
        assert all(isinstance(value, str) and value for value in device.values())


def test_identity_gpu_block_parses_every_queried_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_nvidia_smi(
        monkeypatch,
        devices=(
            "NVIDIA GeForce RTX 5090, GPU-7951f78e, 595.84, 32607\n"
            "NVIDIA A100-SXM4-40GB, GPU-0badc0de, 595.71.05, 40960\n"
        ),
    )
    devices = _identity_without_jax(monkeypatch, "native")["gpu"]
    assert devices == [
        {
            "name": "NVIDIA GeForce RTX 5090",
            "uuid": "GPU-7951f78e",
            "driver_version": "595.84",
            "memory_total_mib": "32607",
        },
        {
            "name": "NVIDIA A100-SXM4-40GB",
            "uuid": "GPU-0badc0de",
            "driver_version": "595.71.05",
            "memory_total_mib": "40960",
        },
    ]


def test_identity_gpu_block_refuses_a_short_query_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_nvidia_smi(monkeypatch, devices="NVIDIA GeForce RTX 5090, GPU-7951f78e\n")
    with pytest.raises(ProbeConventionError, match="nvidia-smi returned 2 fields"):
        _identity_without_jax(monkeypatch, "native")


def test_identity_gpu_processes_track_nvidia_smi_presence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity_without_jax(monkeypatch, "native")
    processes = identity["gpu_compute_processes"]
    if shutil.which("nvidia-smi") is None:
        assert processes is None
    else:
        assert isinstance(processes, list)
        for row in processes:
            assert isinstance(row["pid"], int)
            assert isinstance(row["used_memory_mib"], int)


# ---------------------------------------------------------------------------
# gpu_compute_processes
# ---------------------------------------------------------------------------


def test_gpu_compute_processes_is_none_without_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probe_conventions.shutil, "which", lambda name: None)
    assert gpu_compute_processes() is None


def test_gpu_compute_processes_discards_only_the_named_pids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_nvidia_smi(monkeypatch, compute_apps="4242, 1024\n4243, 2048\n")
    assert gpu_compute_processes() == [
        {"pid": 4242, "used_memory_mib": 1024},
        {"pid": 4243, "used_memory_mib": 2048},
    ]
    # The caller's own legs drop out; whatever is left is foreign compute, and
    # that is the set a probe discards a leg for.
    assert gpu_compute_processes(exclude_pids=[4242]) == [
        {"pid": 4243, "used_memory_mib": 2048}
    ]
    assert gpu_compute_processes(exclude_pids=(4242, 4243)) == []


# ---------------------------------------------------------------------------
# Threading and quiet-gate readbacks
# ---------------------------------------------------------------------------


def test_observed_openmp_threads_is_a_count_or_the_absence_of_openmp() -> None:
    observed = observed_openmp_threads()
    assert observed is None or isinstance(observed, int)
    if observed is not None:
        assert observed >= 1


def test_observed_openmp_threads_is_none_when_libgomp_will_not_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(name: str) -> None:
        raise OSError(f"cannot open shared object file: {name}")

    monkeypatch.setattr(probe_conventions.ctypes, "CDLL", refuse)
    assert observed_openmp_threads() is None


def test_cpu_utilization_delta_is_a_fraction() -> None:
    busy = cpu_utilization_delta(sample_seconds=0.2)
    assert isinstance(busy, float)
    assert 0.0 <= busy <= 1.0


@pytest.mark.parametrize("sample_seconds", [0.0, -1.0])
def test_cpu_utilization_delta_refuses_an_empty_sample(sample_seconds: float) -> None:
    with pytest.raises(ProbeConventionError, match="sample_seconds"):
        cpu_utilization_delta(sample_seconds=sample_seconds)


# ---------------------------------------------------------------------------
# interleave_schedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pair_count", [1, 2, 3, 4, 5, 6])
def test_interleave_is_balanced_and_paired(pair_count: int) -> None:
    schedule = interleave_schedule(pair_count, "native", "jax")
    assert len(schedule) == 2 * pair_count
    assert schedule.count("native") == pair_count
    assert schedule.count("jax") == pair_count
    # Each consecutive slice of two is one complete pair, in either order.
    for index in range(0, len(schedule), 2):
        assert set(schedule[index : index + 2]) == {"native", "jax"}
    # Leading position alternates, so neither lane leads more than one pair
    # more often than the other.
    leaders = schedule[0::2]
    assert leaders == [
        "native" if index % 2 == 0 else "jax" for index in range(pair_count)
    ]
    assert abs(leaders.count("native") - leaders.count("jax")) <= 1


def test_interleave_pattern_is_the_documented_one() -> None:
    assert interleave_schedule(3, "A", "B") == ["A", "B", "B", "A", "A", "B"]


@pytest.mark.parametrize("pair_count", [0, -1])
def test_interleave_refuses_empty_schedules(pair_count: int) -> None:
    with pytest.raises(ProbeConventionError, match="pair_count"):
        interleave_schedule(pair_count, "native", "jax")


def test_interleave_refuses_identical_lanes() -> None:
    with pytest.raises(ProbeConventionError, match="differ"):
        interleave_schedule(3, "native", "native")


# ---------------------------------------------------------------------------
# append_leg_ledger
# ---------------------------------------------------------------------------


def test_ledger_appends_one_canonical_line_per_leg_in_executed_order(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "nested" / "legs.jsonl"
    executed = ["native", "jax", "jax", "native"]
    for index, lane in enumerate(executed):
        append_leg_ledger(ledger, {"leg": index, "lane": lane, "seconds": 1.0 + index})
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(executed)
    assert [json.loads(line)["lane"] for line in lines] == executed
    assert [json.loads(line)["leg"] for line in lines] == [0, 1, 2, 3]
    for line in lines:
        assert line == json.dumps(
            json.loads(line), sort_keys=True, separators=(",", ":")
        )


def test_ledger_never_truncates_an_existing_file(tmp_path: Path) -> None:
    ledger = tmp_path / "legs.jsonl"
    append_leg_ledger(ledger, {"leg": 0})
    first = ledger.read_bytes()
    append_leg_ledger(ledger, {"leg": 1})
    grown = ledger.read_bytes()
    assert grown.startswith(first)
    assert grown[len(first) :] == b'{"leg":1}\n'


def test_ledger_refuses_a_nonfinite_record_and_appends_nothing(tmp_path: Path) -> None:
    ledger = tmp_path / "legs.jsonl"
    append_leg_ledger(ledger, {"leg": 0, "seconds": 1.5})
    before = ledger.read_bytes()
    with pytest.raises(ProbeConventionError, match=str(ledger)):
        append_leg_ledger(ledger, {"leg": 1, "seconds": float("inf")})
    assert ledger.read_bytes() == before


# ---------------------------------------------------------------------------
# write_probe_artifact
# ---------------------------------------------------------------------------


def _payload() -> dict[str, object]:
    return {
        "schema": "probe-demo-v1",
        "identity": {"lane": "native", "hostname": "box"},
        "rows": [{"leg": "native", "seconds": 1.5}],
    }


def test_artifact_is_stamped_canonical_and_newline_terminated(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "probe.json"
    write_probe_artifact(path, _payload())
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    document = json.loads(text)
    assert document["grade"] == PROBE_GRADE
    assert document["schema"] == "probe-demo-v1"
    assert document["identity"] == {"lane": "native", "hostname": "box"}
    assert document["rows"] == [{"leg": "native", "seconds": 1.5}]
    # Canonical encoding: keys sorted, no incidental whitespace.
    assert text == json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    assert list(document) == sorted(document)


def test_artifact_refuses_to_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "probe.json"
    write_probe_artifact(path, _payload())
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_probe_artifact(path, _payload())
    assert path.read_bytes() == before


def test_artifact_requires_a_schema(tmp_path: Path) -> None:
    payload = _payload()
    del payload["schema"]
    with pytest.raises(ProbeConventionError, match="schema"):
        write_probe_artifact(tmp_path / "probe.json", payload)


def test_artifact_requires_an_identity_block(tmp_path: Path) -> None:
    payload = _payload()
    payload["identity"] = "native box"
    with pytest.raises(ProbeConventionError, match="identity"):
        write_probe_artifact(tmp_path / "probe.json", payload)


def test_artifact_refuses_a_caller_supplied_grade(tmp_path: Path) -> None:
    payload = _payload()
    payload["grade"] = "certified"
    with pytest.raises(ProbeConventionError, match="grade"):
        write_probe_artifact(tmp_path / "probe.json", payload)


def test_artifact_validation_refusal_leaves_no_file(tmp_path: Path) -> None:
    # A refusal raised before the encoder runs: nothing was ever opened.
    path = tmp_path / "probe.json"
    payload = _payload()
    del payload["schema"]
    with pytest.raises(ProbeConventionError):
        write_probe_artifact(path, payload)
    assert not path.exists()


def test_artifact_refuses_a_nonfinite_number_buried_in_the_payload(
    tmp_path: Path,
) -> None:
    # The payload passes every field validation; only the encoder can catch
    # this, and it must fail as a typed ProbeConventionError naming the file
    # rather than as a bare ValueError -- and must leave no artifact behind.
    path = tmp_path / "probe.json"
    payload = _payload()
    payload["rows"] = [
        {"leg": "native", "seconds": 1.5},
        {"leg": "jax", "seconds": 0.5, "gradient_inf_norm": float("nan")},
    ]
    with pytest.raises(ProbeConventionError, match=str(path)):
        write_probe_artifact(path, payload)
    assert not path.exists()


# ---------------------------------------------------------------------------
# compile_cache_env
# ---------------------------------------------------------------------------


def test_compile_cache_env_cold_leg_omits_the_variable() -> None:
    assert compile_cache_env(None) == {}


def test_compile_cache_env_warm_leg_pins_an_absolute_path(tmp_path: Path) -> None:
    environment = compile_cache_env(tmp_path / "cache")
    assert list(environment) == [COMPILATION_CACHE_VARIABLE]
    resolved = Path(environment[COMPILATION_CACHE_VARIABLE])
    assert resolved.is_absolute()
    assert resolved == (tmp_path / "cache").resolve()


def test_compile_cache_env_resolves_a_relative_path() -> None:
    environment = compile_cache_env(Path("relative-cache-root"))
    assert Path(environment[COMPILATION_CACHE_VARIABLE]).is_absolute()


# ---------------------------------------------------------------------------
# pinned_environment
# ---------------------------------------------------------------------------

#: A base environment carrying one poisoned variable per scrubbed family that
#: has actually mis-scoped a leg before: a thread re-pin that never touches
#: ``OMP_NUM_THREADS``, a backend reroute, an inherited warm cache, and a
#: device reassignment.
_POISONED_BASE = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/home/probe",
    "LD_LIBRARY_PATH": "/opt/cuda/lib64",
    "OMP_NUM_THREADS": "64",
    "KMP_AFFINITY": "granularity=fine,compact",
    "GOMP_CPU_AFFINITY": "0-63",
    "MKL_NUM_THREADS": "64",
    "OPENBLAS_NUM_THREADS": "64",
    "NUMEXPR_NUM_THREADS": "64",
    "VECLIB_MAXIMUM_THREADS": "64",
    "BLIS_NUM_THREADS": "64",
    "NUMBA_NUM_THREADS": "64",
    "SIMSOPT_BACKEND_MODE": "jax_gpu_fast",
    "JAX_COMPILATION_CACHE_DIR": "/somebody/elses/warm/cache",
    "JAX_PLATFORMS": "cuda",
    "XLA_FLAGS": "--xla_force_host_platform_device_count=8",
    "CUDA_VISIBLE_DEVICES": "3",
    "MPI4PY_RC_INITIALIZE": "true",
}


def test_pin_scrubs_every_poisoned_family_and_keeps_the_loader_paths() -> None:
    environment = pinned_environment(lane="native", omp=8, base=_POISONED_BASE)
    for name in (
        "KMP_AFFINITY",
        "GOMP_CPU_AFFINITY",
        "NUMBA_NUM_THREADS",
        "SIMSOPT_BACKEND_MODE",
        "JAX_COMPILATION_CACHE_DIR",
        "XLA_FLAGS",
        "CUDA_VISIBLE_DEVICES",
    ):
        assert name not in environment, name
    # Non-numerical inheritance survives: the leg still has to load itself.
    assert environment["PATH"] == "/usr/bin:/bin"
    assert environment["HOME"] == "/home/probe"
    assert environment["LD_LIBRARY_PATH"] == "/opt/cuda/lib64"


def test_pin_writes_the_whole_thread_family_at_one_count() -> None:
    environment = pinned_environment(lane="native", omp=8, base=_POISONED_BASE)
    for name in THREAD_COUNT_VARIABLES:
        assert environment[name] == "8", name
    assert environment["OMP_DYNAMIC"] == "FALSE"
    assert environment["OMP_SCHEDULE"] == "STATIC"


def test_pin_always_writes_the_single_rank_and_fp64_pins() -> None:
    for omp in (None, 4):
        environment = pinned_environment(lane="native", omp=omp, base=_POISONED_BASE)
        assert environment["MPI4PY_RC_INITIALIZE"] == "false"
        assert environment["HWLOC_COMPONENTS"] == "-gl"
        assert environment["JAX_ENABLE_X64"] == "1"


def test_pin_without_omp_publishes_the_shipped_default() -> None:
    environment = pinned_environment(lane="native", omp=None, base=_POISONED_BASE)
    # Scrubbed but not replaced: the runtime picks its own thread count, and
    # the inherited 64 is gone either way.
    for name in (*THREAD_COUNT_VARIABLES, "OMP_DYNAMIC", "OMP_SCHEDULE"):
        assert name not in environment, name


def test_pin_omits_the_compilation_cache_unless_it_is_asked_for(
    tmp_path: Path,
) -> None:
    cold = pinned_environment(lane="jax", omp=8, base=_POISONED_BASE)
    # The poisoned base carried a warm cache; a cold leg cannot inherit it.
    assert COMPILATION_CACHE_VARIABLE not in cold
    warm = pinned_environment(
        lane="jax", omp=8, compile_cache_dir=tmp_path / "cache", base=_POISONED_BASE
    )
    assert warm[COMPILATION_CACHE_VARIABLE] == str((tmp_path / "cache").resolve())


def test_pin_sets_jax_platforms_only_when_given() -> None:
    assert "JAX_PLATFORMS" not in pinned_environment(
        lane="jax", omp=None, base=_POISONED_BASE
    )
    native = pinned_environment(
        lane="native", omp=None, jax_platforms="cpu", base=_POISONED_BASE
    )
    assert native["JAX_PLATFORMS"] == "cpu"


def test_pin_defaults_to_the_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROBE_CONVENTIONS_MARKER", "inherited")
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    environment = pinned_environment(lane="native", omp=2)
    assert environment["PROBE_CONVENTIONS_MARKER"] == "inherited"
    assert environment["OMP_NUM_THREADS"] == "2"


def test_pin_honours_an_empty_base_as_an_empty_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROBE_CONVENTIONS_MARKER", "inherited")
    environment = pinned_environment(lane="native", omp=2, base={})
    assert "PROBE_CONVENTIONS_MARKER" not in environment
    assert set(environment) == {
        *THREAD_COUNT_VARIABLES,
        "OMP_DYNAMIC",
        "OMP_SCHEDULE",
        "MPI4PY_RC_INITIALIZE",
        "HWLOC_COMPONENTS",
        "JAX_ENABLE_X64",
    }


@pytest.mark.parametrize("lane", ["", None])
def test_pin_refuses_an_unlabelled_leg(lane: object) -> None:
    with pytest.raises(ProbeConventionError, match="lane label"):
        pinned_environment(lane=lane, omp=8, base=_POISONED_BASE)


@pytest.mark.parametrize("omp", [0, -1])
def test_pin_refuses_a_thread_count_below_one(omp: int) -> None:
    with pytest.raises(ProbeConventionError, match="omp must be >= 1"):
        pinned_environment(lane="native", omp=omp, base=_POISONED_BASE)


# ---------------------------------------------------------------------------
# Shared digest / ULP helpers
# ---------------------------------------------------------------------------


def test_sha256_file_digests_the_exact_bytes(tmp_path: Path) -> None:
    payload = b"probe-conventions digest fixture\n"
    target = tmp_path / "fixture.bin"
    target.write_bytes(payload)
    assert probe_conventions.sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_array_sha256_is_layout_invariant_and_value_sensitive() -> None:
    import numpy as np

    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    c_order = np.ascontiguousarray(values)
    f_order = np.asfortranarray(values)
    assert probe_conventions.array_sha256(c_order) == probe_conventions.array_sha256(
        f_order
    )
    perturbed = c_order.copy()
    perturbed[0, 0] = np.nextafter(perturbed[0, 0], 1.0)
    assert probe_conventions.array_sha256(perturbed) != probe_conventions.array_sha256(
        c_order
    )


def test_ulp_distance_zero_signs_share_a_key() -> None:
    import numpy as np

    distance = probe_conventions.ulp_distance(np.array([-0.0]), np.array([0.0]))
    assert distance.dtype == np.uint64
    assert int(distance[0]) == 0


def test_ulp_distance_counts_adjacent_doubles_across_zero() -> None:
    import numpy as np

    below = np.nextafter(0.0, -1.0)
    above = np.nextafter(0.0, 1.0)
    assert (
        int(probe_conventions.ulp_distance(np.array([0.0]), np.array([above]))[0]) == 1
    )
    assert (
        int(probe_conventions.ulp_distance(np.array([below]), np.array([0.0]))[0]) == 1
    )
    assert (
        int(probe_conventions.ulp_distance(np.array([below]), np.array([above]))[0])
        == 2
    )


def test_ulp_distance_is_exact_beyond_float64_precision() -> None:
    import numpy as np

    # The (-1.0, 1.0) distance is ~2**62.999: far past 2**53, where a float64
    # subtraction of the order keys would round.
    distance = int(probe_conventions.ulp_distance(np.array([-1.0]), np.array([1.0]))[0])
    assert distance == 9214364837600034816
    # One ULP above 1.0 makes the distance odd, so no float64 can hold it:
    # only the uint64 arithmetic reports it exactly.
    odd = int(
        probe_conventions.ulp_distance(
            np.array([-1.0]), np.array([np.nextafter(1.0, 2.0)])
        )[0]
    )
    assert odd == 9214364837600034817
    assert int(float(odd)) != odd
