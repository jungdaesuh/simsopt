"""Shared conventions for the diagnostic A/B probe scripts under ``benchmarks/``.

The probe scripts of the JAX-GPU examples backlog plan
(``docs/jax_gpu_examples_backlog_native_speed_implementation_plan.md``, Phase 0)
are thin: each one builds one family's two lanes and hands the numbers back.
What they must all do identically is the bookkeeping around the numbers —
self-label as diagnostic, alternate the lane order so position bias cannot
masquerade as a ratio, pin the environment each leg runs under, bind each leg
to a runtime identity, record the order the legs actually executed in, and
publish the result once, canonically.  That bookkeeping lives here.

This module is deliberately *conventions only*: it holds no campaign logic and
no gates.  It never decides whether a probe won, never sweeps anything itself,
and never computes a ratio.  A certified native-vs-GPU claim needs its own
preregistered charter (plan §Campaign protocol); nothing published through
:func:`write_probe_artifact` can be one, which is what :data:`PROBE_GRADE`
records in the file itself.

The identity and environment-pinning conventions here are *derived from*
``benchmarks/stage_two_finitebuild_native_gpu.py`` (the certified successor
harness) — same scrub-then-pin discipline, same rationale, several fields in
common — but they are neither a copy of it nor a subset of it.  The identity
published here is a superset (own pid, the compiled extension's digest, the
already-imported library versions, the device inventory); the harness's is a
superset in the other direction (its campaign spec fields).  The pieces are
reimplemented rather than imported because that harness is a 4000-line campaign
instrument whose import pulls in simsopt, numpy, scipy and the compiled
extension, and a probe convention must be importable by anything: everything
below runs on the standard library alone.  In particular this module never
imports JAX; it reports JAX state only when the *calling* process has already
imported it, so that a native leg's identity truthfully records "no JAX here"
instead of creating the very import it is supposed to detect.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every artifact this module writes carries this grade, in the file.  Probes
#: measure; charters certify.
PROBE_GRADE = "diagnostic-not-certifying"

#: The fair-native denominator sweep (plan §Campaign protocol clause 3): the
#: denominator is the swept optimum, never the shipped default.  Probes that
#: quote a native number sweep this set; the constant lives here so no probe
#: re-types it.
OMP_SWEEP = (2, 4, 8, 16, 32, 48)

#: The one environment variable that separates a warm persistent-cache JAX leg
#: from a cold one (``benchmarks/stage_two_finitebuild_native_gpu.py:1189,1286``).
COMPILATION_CACHE_VARIABLE = "JAX_COMPILATION_CACHE_DIR"

# The numerical-threading family, reported by prefix rather than by name: a
# single unscrubbed ``KMP_AFFINITY`` or ``OMP_PROC_BIND`` re-pins a leg without
# touching ``OMP_NUM_THREADS``, so the identity must show the whole family.
# Private and narrower than the public scrub below on purpose — this is what
# the identity's ``threading`` block means by "threading".  It stays importable
# under this name until ``benchmarks/wireframe_gsco_siblings_reference_scale.py``
# migrates to :data:`SCRUBBED_ENVIRONMENT_PREFIXES`, which is that sibling's own
# concatenation, already assembled.
_NUMERICAL_ENVIRONMENT_PREFIXES = (
    "OMP_",
    "GOMP_",
    "KMP_",
    "MKL_",
    "OPENBLAS_",
    "NUMEXPR_",
    "VECLIB_",
    "BLIS_",
    "NUMBA_",
)

#: Everything a leg deletes out of the inherited shell before pinning its own
#: replacements.  Wider than the threading family: one leftover
#: ``JAX_``/``XLA_``/``SIMSOPT_``/``CUDA_`` variable silently re-routes a
#: "native" leg onto the device, into fp32, or onto a warm compilation cache
#: (``benchmarks/stage_two_finitebuild_native_gpu.py:214``), and ``MPI4PY_`` is
#: in the family because the single-rank pin :func:`pinned_environment` writes
#: is only trustworthy if no inherited ``MPI4PY_RC_*`` overrides it.
SCRUBBED_ENVIRONMENT_PREFIXES = _NUMERICAL_ENVIRONMENT_PREFIXES + (
    "JAX_",
    "XLA_",
    "SIMSOPT_",
    "CUDA_",
    "MPI4PY_",
)

#: The thread-count family, pinned as a unit.  Pinning ``OMP_NUM_THREADS``
#: alone leaves the BLAS underneath free to open its own pool, so the leg runs
#: at a thread count nobody chose and no artifact records.
THREAD_COUNT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

#: The quiet-gate sample the 2026-08-16 GSCO campaigns settled on: a 2-second
#: ``/proc/stat`` busy-fraction delta.  The 1-minute loadavg was ruled too slow
#: to notice a neighbour's job starting inside a leg
#: (``docs/receipts/wireframe_gsco_siblings_native_default.md:266``,
#: ``benchmarks/stage_two_finitebuild_native_gpu.py:1348``).
CPU_BUSY_SAMPLE_SECONDS = 2.0

#: The device inventory read back from ``nvidia-smi``, as (query field, record
#: key).  ``--format=...,nounits`` strips the unit off ``memory.total``, which
#: is why the record key carries it instead.
_GPU_QUERY_FIELDS = (
    ("name", "name"),
    ("uuid", "uuid"),
    ("driver_version", "driver_version"),
    ("memory.total", "memory_total_mib"),
)

#: Library versions worth recording, read out of :data:`sys.modules` only.
#: Importing them here would make a native leg's identity lie about what its
#: own process had loaded, and would put numpy/scipy/jaxlib into every probe
#: parent that merely wanted to stamp a receipt.
_OBSERVED_LIBRARIES = ("numpy", "scipy", "jaxlib")

#: The OpenMP runtime everything OpenMP-threaded in this tree links against;
#: ``.so.1`` is the ABI-stable soname.
_LIBGOMP_SONAME = "libgomp.so.1"


class ProbeConventionError(RuntimeError):
    """A probe cannot produce, or cannot publish, trustworthy evidence."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(document: dict[str, object], path: Path) -> str:
    """Sorted, whitespace-free JSON for ``path``, or a typed refusal.

    ``allow_nan=False`` is an integrity rule, not a formatting preference:
    ``NaN``/``Infinity`` are not JSON, and the default encoder would emit bare
    tokens that most readers reject — after the file is on disk, where the
    broken measurement then looks like a published one.  The ``ValueError``
    that refusal raises names neither the file nor the probe, so it is
    re-raised as :class:`ProbeConventionError` carrying the path.  It is never
    swallowed and never downgraded: a nonfinite number in a payload is a
    broken leg, and the probe stops.
    """
    try:
        return json.dumps(
            document, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except ValueError as error:
        raise ProbeConventionError(
            f"probe payload for {path} is not canonical JSON: {error}"
        ) from error


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _git_identity() -> dict[str, object]:
    """Commit, porcelain status, and a content digest of the dirtied files.

    ``-z`` is the machine-parsing status format: NUL-separated fields, no
    quoting, and rename/copy entries emit the new path first followed by the
    original path as its own field.  Splitting on lines instead would drop
    paths containing spaces or non-ASCII bytes and silently under-report a
    dirty tree — which is the one thing this field exists to catch.  ``-uall``
    enumerates untracked *directories* file by file; plain ``-u`` collapses one
    to a single ``?? dir/`` entry, which is not a path this can hash, so an
    entire untracked probe or harness would be recorded as one unhashed line.

    Digested: every status path that is a file on disk at capture time, hashed
    by its own content.  Not digested: paths the status reports as deleted, and
    the pre-rename original of a rename — they have no bytes left to hash and
    are recorded by their status entry alone, so a receipt reader sees the
    deletion without being handed a digest that does not exist.
    """
    fields = _git("status", "--porcelain", "-uall", "-z").split("\0")
    if fields and fields[-1] == "":
        fields.pop()
    status_entries: list[str] = []
    changed_hashes: dict[str, str] = {}
    index = 0
    while index < len(fields):
        entry = fields[index]
        index += 1
        status_entries.append(entry)
        path_text = entry[3:]
        if entry[:1] in {"R", "C"} and index < len(fields):
            status_entries.append(fields[index])
            index += 1
        candidate = REPO_ROOT / path_text
        if candidate.is_file():
            changed_hashes[path_text] = sha256_file(candidate)
    return {
        "commit": _git("rev-parse", "HEAD").strip(),
        "status": status_entries,
        "changed_file_sha256": changed_hashes,
    }


def _threading_identity() -> dict[str, object]:
    """Every numerical-threading variable actually visible to this process.

    This is the *request* side only.  What the OpenMP runtime went on to do
    with it is :func:`observed_openmp_threads`, which the caller takes inside
    the leg where libgomp is loaded.
    """
    return {
        "environment": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith(_NUMERICAL_ENVIRONMENT_PREFIXES)
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_count": os.cpu_count(),
    }


def _simsoptpp_identity() -> dict[str, str] | None:
    """Path and content digest of the compiled extension, without importing it.

    ``find_spec`` locates a top-level extension by walking the same path the
    import system would, and returns without executing it — which matters
    twice over: importing ``simsoptpp`` costs seconds and, in a native leg,
    drags in the jax-jitted ``simsopt.geo`` objectives this module exists to
    stay clear of.  The digest is what makes a receipt reproducible: the same
    commit can be built into different extensions, and only the ``.so``'s bytes
    say which one ran.  ``None`` means no extension is importable from this
    process at all — a build fact, distinct from a digest.
    """
    spec = importlib.util.find_spec("simsoptpp")
    if spec is None or spec.origin is None:
        return None
    origin = Path(spec.origin)
    if not origin.is_file():
        return None
    return {"path": str(origin), "sha256": sha256_file(origin)}


def _gpu_identity() -> list[dict[str, str]] | None:
    """Every device on the box, or ``None`` off GPU hosts.

    Recorded for both lanes: a native denominator measured on a different box —
    or on the same box after a driver bump — is not the denominator a ratio
    was formed against, and the driver version is the field that says so.
    """
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    query = ",".join(field for field, _ in _GPU_QUERY_FIELDS)
    completed = subprocess.run(
        [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    devices: list[dict[str, str]] = []
    for line in completed.stdout.strip().splitlines():
        values = [field.strip() for field in line.split(",")]
        if len(values) != len(_GPU_QUERY_FIELDS):
            raise ProbeConventionError(
                f"nvidia-smi returned {len(values)} fields for {query!r}: {line!r}"
            )
        devices.append(
            {key: value for (_, key), value in zip(_GPU_QUERY_FIELDS, values)}
        )
    return devices


def gpu_compute_processes(
    exclude_pids: Iterable[int] = (),
) -> list[dict[str, int]] | None:
    """Compute apps holding the GPU right now, or ``None`` off GPU hosts.

    Lane-agnostic on purpose: a *native* leg sharing the box with somebody
    else's GPU compute is as compromised a measurement as a GPU leg would be.
    ``None`` means the host has no ``nvidia-smi`` at all — a different fact
    from "no processes", and the two must not encode the same.  Where
    ``nvidia-smi`` does exist, a failing query is fatal rather than empty.

    ``exclude_pids`` drops the caller's *own* known legs, so what remains is
    exactly the foreign compute a probe discards a leg for.  The caller owns
    that list because only it knows which pids it launched; the identity block
    records the unfiltered set plus ``own_pid``, so a reader can redo the
    subtraction from the artifact alone.
    """
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    excluded = set(exclude_pids)
    completed = subprocess.run(
        [
            executable,
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, int]] = []
    for line in completed.stdout.strip().splitlines():
        pid_text, memory_text = (field.strip() for field in line.split(","))
        pid = int(pid_text)
        if pid in excluded:
            continue
        rows.append({"pid": pid, "used_memory_mib": int(memory_text)})
    return rows


def runtime_identity(lane: str) -> dict[str, object]:
    """Bind one leg to the tree, the build, the host, the box state and the clock.

    ``lane`` is the caller's label for the leg (``"native"`` / ``"jax"`` in the
    families this plan probes); it is recorded verbatim and interpreted
    nowhere, since lane *policy* is charter business.

    JAX fields appear if and only if the calling process has already imported
    JAX — read out of :data:`sys.modules`, never imported here.  That is not a
    convenience: ``simsopt.geo`` objectives are jax-jitted, so a native leg can
    import JAX transitively and quietly initialize CUDA or run float32, and
    both of those make the "native" number something other than a native
    measurement.  The identity has to be able to say so.  ``numpy``, ``scipy``
    and ``jaxlib`` follow the same read-only rule: present when the process
    already loaded them, absent otherwise, never imported to fill a field in.
    """
    identity: dict[str, object] = {
        "lane": lane,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "own_pid": os.getpid(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "git": _git_identity(),
        "simsoptpp": _simsoptpp_identity(),
        "threading": _threading_identity(),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "timestamp_ns": time.time_ns(),
        "wallclock_utc": datetime.now(timezone.utc).isoformat(),
        "loadavg": list(os.getloadavg()),
    }
    for name in _OBSERVED_LIBRARIES:
        module = sys.modules.get(name)
        if module is not None:
            identity[f"{name}_version"] = str(module.__version__)
    jax_imported = "jax" in sys.modules
    identity["jax_imported"] = jax_imported
    if jax_imported:
        jax = sys.modules["jax"]
        identity["jax_version"] = str(jax.__version__)
        identity["jax_default_backend"] = str(jax.default_backend())
        identity["jax_enable_x64"] = bool(jax.config.read("jax_enable_x64"))
        identity["jax_devices"] = [
            {"platform": str(device.platform), "kind": str(device.device_kind)}
            for device in jax.local_devices()
        ]
    identity["gpu"] = _gpu_identity()
    identity["gpu_compute_processes"] = gpu_compute_processes()
    return identity


def observed_openmp_threads() -> int | None:
    """What the OpenMP runtime reports, as opposed to what the pin requested.

    ``OMP_NUM_THREADS`` is a request.  A leg can be re-pinned underneath it by
    ``OMP_THREAD_LIMIT``, by a library calling ``omp_set_num_threads``, or by a
    cgroup CPU quota, and then the identity's environment block records a
    number the run never used and the fair-native denominator is quoted at the
    wrong thread count.  This is the readback: ``omp_get_max_threads`` out of
    libgomp itself, the same ctypes call the fair-bar provenance shim takes
    (``benchmarks/genuine_675_fair_bar.py:387``).  The libgomp already mapped
    into this process is preferred, so a leg reads back the runtime it is
    actually using rather than whichever one the loader would pick.

    ``None`` means libgomp could not be loaded at all — a host fact, and a
    distinct one.  It must never be folded into a thread count: "no OpenMP
    here" and "one thread" are different measurements.
    """
    mapped: str | None = None
    with open("/proc/self/maps", encoding="ascii") as handle:
        for line in handle:
            if "libgomp" in line:
                mapped = line.rsplit(" ", 1)[-1].strip()
                break
    try:
        library = ctypes.CDLL(mapped if mapped is not None else _LIBGOMP_SONAME)
        return int(library.omp_get_max_threads())
    except (OSError, AttributeError):
        return None


def _proc_stat_totals() -> tuple[int, int]:
    """Aggregate (total jiffies, idle jiffies) from the ``cpu`` summary line."""
    with open("/proc/stat", encoding="ascii") as handle:
        fields = [int(value) for value in handle.readline().split()[1:]]
    return sum(fields), fields[3] + fields[4]


def cpu_utilization_delta(sample_seconds: float = CPU_BUSY_SAMPLE_SECONDS) -> float:
    """Busy fraction of the whole box across ``sample_seconds``, in ``[0, 1]``.

    The quiet-gate signal.  A timed native leg shares the box with everything
    else on it, so a neighbour's job is a ratio-sized error; the 1-minute
    loadavg is too slow to see one start, which is why the certified harness
    gates on this delta instead
    (``benchmarks/stage_two_finitebuild_native_gpu.py:1348``).  ``idle`` here is
    idle plus iowait, so a leg blocked on disk counts as busy — it is competing
    for the box either way.

    This returns the number; the threshold is the caller's charter, not a
    convention.  A ``/proc/stat`` that has not advanced is a broken sample and
    fails loudly rather than reporting a plausible ``0.0``.
    """
    if sample_seconds <= 0.0:
        raise ProbeConventionError(
            f"sample_seconds must be > 0, got {sample_seconds!r}"
        )
    total_before, idle_before = _proc_stat_totals()
    time.sleep(sample_seconds)
    total_after, idle_after = _proc_stat_totals()
    total_delta = total_after - total_before
    if total_delta <= 0:
        raise ProbeConventionError(
            f"/proc/stat did not advance across a {sample_seconds}s sample"
        )
    return 1.0 - (idle_after - idle_before) / total_delta


def interleave_schedule(pair_count: int, first: str, second: str) -> list[str]:
    """Leg order for ``pair_count`` A/B pairs, alternating which lane leads.

    Convention (one of the two the plan allows, fixed here so every probe uses
    the same one): each pair runs both lanes back to back, and consecutive
    pairs swap which lane goes first — ``A,B, B,A, A,B, ...``.  Both lanes
    therefore run exactly ``pair_count`` times, and neither leads more than one
    pair more often than the other, so a monotone drift in the box (thermal,
    a neighbour's job starting) cannot land systematically on one lane.

    This is the order a probe *intends*.  The order it got is
    :func:`append_leg_ledger`.
    """
    if pair_count < 1:
        raise ProbeConventionError(f"pair_count must be >= 1, got {pair_count}")
    if first == second:
        raise ProbeConventionError(
            f"interleaved lanes must differ, got {first!r} twice"
        )
    schedule: list[str] = []
    for index in range(pair_count):
        leading, trailing = (first, second) if index % 2 == 0 else (second, first)
        schedule.append(leading)
        schedule.append(trailing)
    return schedule


def array_sha256(array: object) -> str:
    """SHA-256 of an array's C-contiguous raw bytes (dtype/shape not encoded).

    The digest is over ``np.ascontiguousarray(array).tobytes()`` alone so two
    lanes that produce the same fp64 values hash equal regardless of layout.
    Callers that need dtype/shape identity must record them beside the digest.
    NumPy is imported inside the function on purpose: this module must stay
    importable before a probe pins its numerical environment, and importing
    NumPy at module scope would initialize threading ahead of the pin.
    """
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def ulp_distance(first: object, second: object) -> object:
    """Exact elementwise ULP distance between two float64 arrays (uint64).

    Maps each float64 onto the order-preserving integer key (sign-folded
    two's-complement trick, ``-0.0`` and ``+0.0`` share a key), then returns
    ``|key(a) - key(b)|`` as ``uint64`` — exact for every representable pair,
    unlike a float64 subtraction of the keys.  NaNs participate as their bit
    patterns; callers gate finiteness separately.  Deferred NumPy import for
    the same environment-pinning reason as :func:`array_sha256`.
    """
    import numpy as np

    def biased_keys(values: object) -> object:
        bits = np.ascontiguousarray(values, dtype=np.float64).view(np.int64)
        ordered = np.where(bits < 0, np.int64(-0x8000000000000000) - bits, bits)
        # Shift the monotone int64 key into uint64 by the sign bias, so the
        # subtraction below is exact for negative keys too (a bare
        # ``astype(uint64)`` wraps them to huge values -- caught by the unit
        # tests in tests/benchmarks/test_probe_conventions.py).
        return ordered.astype(np.uint64) + np.uint64(0x8000000000000000)

    a64 = biased_keys(first)
    b64 = biased_keys(second)
    return np.where(a64 >= b64, a64 - b64, b64 - a64)


def append_leg_ledger(ledger_path: Path, record: Mapping[str, object]) -> None:
    """Append one leg to a JSONL ledger: one canonical line, flushed and fsynced.

    Executed order is evidence, and it is the evidence a single artifact
    written at the end cannot carry.  Legs run in separate processes and often
    separate interpreters; each appends its own line as it finishes, so an
    interleave that degenerated — a leg that crashed, a lane retried, two
    probes sharing a run directory — is visible in the file afterwards instead
    of assumed away by :func:`interleave_schedule`'s intended order.

    Append-only by construction: the file is opened ``"a"``, never truncated
    and never rewritten, so a probe cannot revise its own history.  Each line
    is written in one ``write`` call on an ``O_APPEND`` descriptor — that, not
    the ``fsync``, is what keeps concurrent appenders from interleaving within
    a line (POSIX guarantees atomic append only up to ``PIPE_BUF``-scale
    writes; a pathologically long record could still tear).  The ``fsync`` is
    for durability: a leg that returned is on disk even if the box dies.
    """
    encoded = _canonical_json(dict(record), ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_probe_artifact(path: Path, payload: dict[str, object]) -> None:
    """Publish one probe artifact: grade-stamped, canonical, written once.

    The caller owns the payload — its ``schema`` name and its ``identity``
    block from :func:`runtime_identity`, both required, because an artifact
    without a schema cannot be read back and one without provenance cannot be
    trusted.  This function owns the grade stamp, so ``grade`` in the payload
    is a conflict, not an override.

    An existing file is never overwritten: a probe that would clobber its own
    evidence fails with ``FileExistsError`` and the earlier run survives.  A
    payload carrying a nonfinite number fails with
    :class:`ProbeConventionError` naming this path, and nothing is written.
    """
    schema = payload.get("schema")
    if not isinstance(schema, str) or not schema:
        raise ProbeConventionError("probe payload must carry a non-empty 'schema'")
    if not isinstance(payload.get("identity"), Mapping):
        raise ProbeConventionError(
            "probe payload must embed an 'identity' mapping from runtime_identity()"
        )
    if "grade" in payload:
        raise ProbeConventionError(
            f"'grade' is stamped by write_probe_artifact ({PROBE_GRADE}), "
            "not supplied by the payload"
        )
    encoded = _canonical_json({**payload, "grade": PROBE_GRADE}, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(encoded + "\n")


def compile_cache_env(cache_dir: Path | None) -> dict[str, str]:
    """Environment additions selecting a warm persistent-cache leg, or a cold one.

    ``None`` is the cold leg and returns ``{}``, and that is an *addition* of
    nothing — not a guarantee of absence.  Merged onto an unscrubbed parent
    environment (``os.environ | compile_cache_env(None)``) the cold leg
    inherits whatever cache the shell exported and is quietly warm, which is
    the one confusion warm-vs-cold scoping exists to prevent.  Removing the
    variable is therefore the caller's obligation, and
    :func:`pinned_environment` is the sanctioned way to discharge it: it scrubs
    the whole ``JAX_`` family before pinning, so a cold leg built there cannot
    inherit a cache by construction.  A path is the warm leg and is resolved to
    an absolute path, since the leg usually runs as a child process whose
    working directory is not the caller's.

    Warm and cold are separate scopes and never fold into one number
    (plan §Campaign protocol clause 4); this returns the lever, not the policy.
    """
    if cache_dir is None:
        return {}
    return {COMPILATION_CACHE_VARIABLE: str(cache_dir.resolve())}


def pinned_environment(
    *,
    lane: str,
    omp: int | None,
    jax_platforms: str | None = None,
    compile_cache_dir: Path | None = None,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment one leg runs under: inherited shell, scrubbed, then pinned.

    Scrub-then-pin, in that order, is the whole design.  Building a leg
    environment by updating ``os.environ`` leaves every inherited selector the
    caller did not think to name — and the ones that matter are exactly the
    ones nobody names: ``KMP_AFFINITY`` re-pins threads without touching
    ``OMP_NUM_THREADS``, ``JAX_COMPILATION_CACHE_DIR`` turns a cold leg warm,
    ``CUDA_VISIBLE_DEVICES`` moves a device leg to another card,
    ``SIMSOPT_BACKEND_MODE`` reroutes a native leg to the GPU backend.  Every
    prefix in :data:`SCRUBBED_ENVIRONMENT_PREFIXES` is deleted first, so a
    variable is present in the result if and only if this function put it
    there.  Non-numerical inheritance (``PATH``, ``LD_LIBRARY_PATH``,
    ``PYTHONPATH``, ``HOME``) survives: the leg still has to be able to load
    its own interpreter and libraries.

    ``lane`` is the caller's label for the leg and must match the one it hands
    :func:`runtime_identity`, so the pin and the identity in one artifact
    describe the same thing; it is validated and named in refusals but never
    consulted for policy.  This function deliberately does not decide that
    "native" means ``JAX_PLATFORMS=cpu`` or that "jax" means a warm cache —
    those are charter decisions, passed in explicitly.

    ``omp=None`` publishes the shipped default: the inherited threading
    configuration is still scrubbed, but nothing is pinned in its place, so the
    runtime picks its own thread count.  Only a disclosure lane wants that; a
    timed lane pins, and pins the whole family at once
    (:data:`THREAD_COUNT_VARIABLES`), since ``OMP_NUM_THREADS`` alone leaves
    the BLAS free to open a second pool.  ``OMP_DYNAMIC=FALSE`` stops the
    runtime handing back fewer threads than asked for, and
    ``OMP_SCHEDULE=STATIC`` fixes the loop partitioning so two legs at the same
    thread count do the same work in the same order.

    ``compile_cache_dir=None`` is a cold leg, and here it genuinely cannot
    inherit a warm one: the scrub removed ``JAX_COMPILATION_CACHE_DIR`` before
    anything was pinned, and only ``compile_cache_dir`` can put it back.

    ``base`` replaces the inherited environment for testing and for hermetic
    legs; an empty mapping is honoured as an empty base, never silently
    swapped for the caller's own ``os.environ``.
    """
    if not isinstance(lane, str) or not lane:
        raise ProbeConventionError(
            f"pinned_environment requires a non-empty lane label, got {lane!r}"
        )
    if omp is not None and omp < 1:
        raise ProbeConventionError(f"lane {lane!r}: omp must be >= 1, got {omp!r}")
    source = os.environ if base is None else base
    environment = {
        name: value
        for name, value in dict(source).items()
        if not name.startswith(SCRUBBED_ENVIRONMENT_PREFIXES)
    }
    environment.update(
        {
            # Every leg is single-rank by design; with this pin MPI_Init never
            # runs and any future communicator use aborts loudly.
            "MPI4PY_RC_INITIALIZE": "false",
            # Dormant while the mpi4py pin is set, load-bearing the moment a
            # future MPI leg drops it: during MPI_Init hwloc's ``gl`` plugin
            # connects to X displays, and a box with a wedged Xwayland listener
            # blocks such connects forever (measured 2026-08-17), a hang a
            # wall-clock leg would misreport as a timeout.
            "HWLOC_COMPONENTS": "-gl",
            # ``simsopt.geo`` objectives are jax-jitted, so even a native leg
            # imports JAX transitively, and that import defaults to float32
            # while the scrub above removed any inherited x64 setting.
            # Unpinned, the jax-jitted pieces evaluate in fp32: measured
            # 2026-08-17 the native gradient forked from the GPU lane by up to
            # 2.6e-6 per component while J moved 3e-13, and FD arbitration
            # convicted the native value
            # (docs/jax_gpu_finitebuild_fp64_taint_diagnostic.md).  fp64 is
            # part of the physics specification, not a tuning knob.
            "JAX_ENABLE_X64": "1",
        }
    )
    if omp is not None:
        thread_count = str(omp)
        environment.update({name: thread_count for name in THREAD_COUNT_VARIABLES})
        environment["OMP_DYNAMIC"] = "FALSE"
        environment["OMP_SCHEDULE"] = "STATIC"
    if jax_platforms is not None:
        # ``cpu`` keeps a native leg's transitive JAX import off the device, so
        # it cannot put a second CUDA context on the card a GPU leg is timed on.
        environment["JAX_PLATFORMS"] = jax_platforms
    environment.update(compile_cache_env(compile_cache_dir))
    return environment
