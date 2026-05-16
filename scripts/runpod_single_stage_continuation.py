#!/usr/bin/env python3
"""Launch the single-stage continuation workflow on a Runpod GPU pod."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tempfile
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_COLUMBIA_ROOT = REPO_ROOT.parent
DEFAULT_REMOTE_COLUMBIA_ROOT = PurePosixPath("/workspace/columbia")
DEFAULT_REMOTE_OUTPUT_ROOT = PurePosixPath("/workspace/continuation-runs")
DEFAULT_REMOTE_CACHE_DIR = PurePosixPath("/workspace/.jax-cache/single-stage-continuation")
DEFAULT_REMOTE_CONDA_ROOT = PurePosixPath("/opt/conda")
DEFAULT_ENV_NAME = "jax"
DEFAULT_SYSTEM_DEPS_MODE = "auto"
EXACT_JAX_GPU_WHEEL_SPEC = "jax[cuda12]==0.9.2"
REQUIRED_RUNPOD_CUDA_TOOLKIT_RELEASE = "12.9"
DEFAULT_PLASMA_SURF_FILENAME = "wout_nfp5ginsburg_desc_iota21.nc"
DEFAULT_LOCAL_EQUILIBRIUM_PATH = (
    LOCAL_COLUMBIA_ROOT
    / "DATABASE"
    / "EQUILIBRIA"
    / DEFAULT_PLASMA_SURF_FILENAME
)
DEFAULT_LOCAL_DONOR_RUN_DIR = (
    LOCAL_COLUMBIA_ROOT
    / "autoresearch"
    / "single_stage_results"
    / f"outputs-{DEFAULT_PLASMA_SURF_FILENAME}"
    / "mpol=8-ntor=6-d3476e33-1775578133118"
)
DEFAULT_LOCAL_OUTPUT_ROOT = REPO_ROOT / ".artifacts" / "runpod_single_stage_continuation"
DEFAULT_POD_ID = os.environ.get("SIMSOPT_RUNPOD_POD_ID")
DEFAULT_BACKEND_MODE = "jax_gpu_parity"
DEFAULT_TRANSFER_GUARD = "disallow"
_RUNPOD_INFO_COMMAND = ("runpodctl", "ssh", "info")
_CANDIDATE_LEDGER_SCRIPT = (
    REPO_ROOT / "examples" / "single_stage_optimization" / "candidate_ledger.py"
)
_REMOTE_REPO_BUILD_CACHE_PREFIX = ".runpod-build-cache-"
_PORTABLE_TAR_SCRIPT = REPO_ROOT / "scripts" / "portable_tar.sh"
_REQUIRED_DONOR_FILENAMES = (
    "results.json",
    "biot_savart_opt.json",
    "surf_opt.json",
)
_JAX_RUNTIME_SPEC_FILENAME = "single_stage_jax_runtime_spec.json"
_TAR_EXCLUDES = (
    ".git",
    ".DS_Store",
    "._*",
    ".venv-simsopt-jax",
    ".venv-local",
    ".conda",
    ".miniforge",
    ".tmp",
    "build",
    ".artifacts",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "__pycache__",
    "consolidated_code.txt",
)


@dataclass(frozen=True)
class SshInfo:
    host: str
    port: int
    key_path: Path
    user: str
    pod_id: str | None

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


@dataclass(frozen=True)
class LaunchPlan:
    pod_id: str | None
    run_id: str
    pretend_version: str
    local_columbia_root: Path
    local_repo_root: Path
    local_equilibrium_path: Path
    local_donor_run_dir: Path
    local_output_dir: Path
    remote_columbia_root: str
    remote_repo_root: str
    remote_equilibrium_path: str
    remote_donor_run_dir: str
    remote_output_root: str
    remote_run_root: str
    remote_jax_profile_dir: str | None
    remote_summary_path: str
    remote_validation_path: str | None
    remote_cache_dir: str
    remote_conda_root: str
    plasma_surf_filename: str
    system_deps_mode: str
    backend_mode: str
    transfer_guard: str
    trial_policy: str
    run_mode: str
    max_final_field_error: float | None
    max_final_abs_iota_error: float | None
    max_final_non_qs: float | None
    mpol: int
    ntor: int
    nphi: int
    ntheta: int
    maxiter: int
    coarse_maxiter: int
    medium_maxiter: int
    prefinal_maxiter: int
    fetch_full_run_root: bool
    single_stage_passthrough_args: tuple[str, ...] = ()
    local_donor_run_dirs: tuple[Path, ...] = ()
    remote_donor_run_dirs: tuple[str, ...] = ()


@dataclass(frozen=True)
class FetchArtifact:
    label: str
    remote_source: PurePosixPath
    local_destination: Path
    recursive: bool
    required: bool


@dataclass(frozen=True)
class FetchReport:
    fetched_paths: tuple[str, ...]
    required_failures: tuple[str, ...]
    optional_failures: tuple[str, ...]

    @property
    def has_required_failures(self) -> bool:
        return bool(self.required_failures)

    @property
    def has_optional_failures(self) -> bool:
        return bool(self.optional_failures)


def local_donor_run_dirs(plan: LaunchPlan) -> tuple[Path, ...]:
    if plan.local_donor_run_dirs:
        return plan.local_donor_run_dirs
    return (plan.local_donor_run_dir,)


def remote_donor_run_dirs(plan: LaunchPlan) -> tuple[str, ...]:
    if plan.remote_donor_run_dirs:
        return plan.remote_donor_run_dirs
    return (plan.remote_donor_run_dir,)


def _normalize_passthrough_args(raw_args: list[str]) -> tuple[str, ...]:
    return tuple(token for token in raw_args if token != "--")


def launch_plan_uses_campaign(plan: LaunchPlan) -> bool:
    return len(local_donor_run_dirs(plan)) > 1


def resolve_remote_profiling_report_path(plan: LaunchPlan) -> PurePosixPath:
    report_name = (
        "campaign_profiling_report.md"
        if launch_plan_uses_campaign(plan)
        else "continuation_profiling_report.md"
    )
    return PurePosixPath(plan.remote_run_root) / report_name


def _run_id_now() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _posix(path: PurePosixPath | str) -> str:
    return str(path)


def _require_existing_path(path: Path, *, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"{description} does not exist: {resolved}")
    return resolved


def _derive_pretend_version(local_repo_root: Path) -> str:
    head = subprocess.run(
        ["git", "-C", str(local_repo_root), "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if head.returncode != 0:
        stderr = head.stderr.strip()
        raise SystemExit(
            stderr or f"Could not resolve git HEAD for {local_repo_root}"
        )
    short_head = head.stdout.strip()
    if not short_head:
        raise SystemExit(f"Could not resolve git HEAD for {local_repo_root}")
    return f"0.0.dev0+g{short_head}"


def _require_donor_artifacts(run_dir: Path) -> Path:
    resolved = _require_existing_path(run_dir, description="donor run directory")
    has_legacy_seed = all(
        (resolved / filename).exists()
        for filename in ("biot_savart_opt.json", "surf_opt.json")
    )
    has_jax_runtime_seed = (resolved / _JAX_RUNTIME_SPEC_FILENAME).exists()
    missing = ["results.json"] if not (resolved / "results.json").exists() else []
    if not has_legacy_seed and not has_jax_runtime_seed:
        missing.extend(
            filename
            for filename in _REQUIRED_DONOR_FILENAMES[1:]
            if not (resolved / filename).exists()
        )
        missing.append(_JAX_RUNTIME_SPEC_FILENAME)
    if missing:
        raise SystemExit(
            "donor run directory is missing required artifacts: "
            + ", ".join(str(resolved / filename) for filename in missing)
        )
    return resolved


def _relative_to_root(path: Path, root: Path, *, description: str) -> Path:
    try:
        return path.relative_to(root)
    except ValueError as exc:
        raise SystemExit(
            f"{description} must stay under {root}, got {path}"
        ) from exc


def map_local_path_to_remote(
    local_path: Path,
    *,
    local_columbia_root: Path,
    remote_columbia_root: PurePosixPath,
) -> PurePosixPath:
    relative = _relative_to_root(
        local_path,
        local_columbia_root,
        description="local path",
    )
    remote_path = remote_columbia_root
    for part in relative.parts:
        remote_path /= part
    return remote_path


def resolve_remote_jax_profile_dir(
    *,
    remote_run_root: PurePosixPath,
    requested_profile_dir: str | None,
) -> PurePosixPath | None:
    if requested_profile_dir is None:
        return None
    requested_path = PurePosixPath(requested_profile_dir)
    if requested_path.is_absolute():
        return requested_path
    return remote_run_root / requested_path


def parse_runpod_ssh_info(payload: dict[str, object], *, user: str) -> SshInfo:
    host = payload.get("ip")
    port = payload.get("port")
    key_payload = payload.get("ssh_key")
    if not isinstance(host, str) or not host:
        raise SystemExit(f"Runpod ssh info missing ip: {payload}")
    if not isinstance(port, int):
        raise SystemExit(f"Runpod ssh info missing port: {payload}")
    if not isinstance(key_payload, dict):
        raise SystemExit(f"Runpod ssh info missing ssh_key payload: {payload}")
    key_path = key_payload.get("path")
    if not isinstance(key_path, str) or not key_path:
        raise SystemExit(f"Runpod ssh info missing ssh key path: {payload}")
    pod_id = payload.get("id")
    return SshInfo(
        host=host,
        port=port,
        key_path=Path(key_path).expanduser().resolve(),
        user=user,
        pod_id=pod_id if isinstance(pod_id, str) else None,
    )


def resolve_ssh_info(args: argparse.Namespace) -> SshInfo:
    if args.host is not None and args.port is not None and args.ssh_key is not None:
        return SshInfo(
            host=args.host,
            port=int(args.port),
            key_path=Path(args.ssh_key).expanduser().resolve(),
            user=args.ssh_user,
            pod_id=args.pod_id,
        )
    if args.pod_id is None:
        raise SystemExit(
            "Pass --pod-id or provide --host, --port, and --ssh-key explicitly."
        )
    completed = subprocess.run(
        [*_RUNPOD_INFO_COMMAND, args.pod_id],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise SystemExit(
            stderr
            or f"runpodctl ssh info failed for pod {args.pod_id}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Could not parse runpodctl ssh info output for pod {args.pod_id}: {exc}"
        ) from exc
    return parse_runpod_ssh_info(payload, user=args.ssh_user)


def ssh_base_command(ssh_info: SshInfo) -> list[str]:
    return [
        "ssh",
        "-i",
        str(ssh_info.key_path),
        "-p",
        str(ssh_info.port),
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]


def scp_base_command(ssh_info: SshInfo) -> list[str]:
    return [
        "scp",
        "-i",
        str(ssh_info.key_path),
        "-P",
        str(ssh_info.port),
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]


def build_repo_tar_command(
    *,
    local_columbia_root: Path,
    repo_relative_path: Path,
    output_path: Path | None = None,
) -> list[str]:
    command = [
        "bash",
        str(_PORTABLE_TAR_SCRIPT),
        "--root",
        str(local_columbia_root),
    ]
    if output_path is None:
        command.append("--gzip")
    else:
        command.extend(["--file", str(output_path), "--gzip"])
    for pattern in _TAR_EXCLUDES:
        command.extend(["--exclude", pattern])
    command.append(str(repo_relative_path))
    return command


def build_repo_tar_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("COPYFILE_DISABLE", "1")
    env.setdefault("COPY_EXTENDED_ATTRIBUTES_DISABLE", "1")
    return env


def build_remote_repo_extract_command(
    plan: LaunchPlan,
    ssh_info: SshInfo,
    *,
    remote_archive_path: PurePosixPath,
) -> list[str]:
    repo_root = PurePosixPath(plan.remote_repo_root)
    build_cache_root = PurePosixPath(plan.remote_columbia_root) / (
        f"{_REMOTE_REPO_BUILD_CACHE_PREFIX}{repo_root.name}"
    )
    quoted_columbia_root = shlex.quote(plan.remote_columbia_root)
    quoted_repo_root = shlex.quote(plan.remote_repo_root)
    quoted_build_cache_root = shlex.quote(str(build_cache_root))
    quoted_remote_archive_path = shlex.quote(str(remote_archive_path))
    remote_command = (
        f"mkdir -p {quoted_columbia_root} && "
        f"if [ -d {quoted_repo_root}/build ]; then "
        f"rm -rf {quoted_build_cache_root} && "
        f"mv {quoted_repo_root}/build {quoted_build_cache_root}; "
        f"fi && "
        f"rm -rf {quoted_repo_root} && "
        f"trap 'rm -f {quoted_remote_archive_path}' EXIT && "
        f"tar --no-same-owner --no-same-permissions -xzf {quoted_remote_archive_path} -C {quoted_columbia_root} && "
        f"if [ -d {quoted_build_cache_root} ]; then "
        f"mv {quoted_build_cache_root} {quoted_repo_root}/build; "
        f"fi"
    )
    return [
        *ssh_base_command(ssh_info),
        ssh_info.target,
        f"bash -lc {shlex.quote(remote_command)}",
    ]


def remote_repo_archive_path(plan: LaunchPlan) -> PurePosixPath:
    repo_relative_path = _relative_to_root(
        plan.local_repo_root,
        plan.local_columbia_root,
        description="local repo root",
    )
    return PurePosixPath("/tmp") / f"{repo_relative_path.name}-{plan.run_id}.tar.gz"


def build_remote_prepare_script(plan: LaunchPlan) -> str:
    lines = [
        "set -euxo pipefail",
        f"mkdir -p {shlex.quote(plan.remote_columbia_root)}",
        f"mkdir -p {shlex.quote(str(PurePosixPath(plan.remote_equilibrium_path).parent))}",
        f"mkdir -p {shlex.quote(plan.remote_output_root)}",
        f"mkdir -p {shlex.quote(plan.remote_cache_dir)}",
    ]
    donor_parent_dirs = sorted(
        {str(PurePosixPath(path).parent) for path in remote_donor_run_dirs(plan)}
    )
    for donor_parent_dir in donor_parent_dirs:
        lines.append(f"mkdir -p {shlex.quote(donor_parent_dir)}")
    if plan.remote_jax_profile_dir is not None:
        lines.append(f"mkdir -p {shlex.quote(plan.remote_jax_profile_dir)}")
    return "\n".join(lines)


def build_remote_execution_script(plan: LaunchPlan) -> str:
    repo_root = shlex.quote(plan.remote_repo_root)
    conda_root = shlex.quote(plan.remote_conda_root)
    env_name = shlex.quote(DEFAULT_ENV_NAME)
    run_root = shlex.quote(plan.remote_run_root)
    run_lock_dir = shlex.quote(f"{plan.remote_run_root}.lock")
    lines = [
        "set -euxo pipefail",
        f'SYSTEM_DEPS_MODE={shlex.quote(plan.system_deps_mode)}',
        f'REPO_ROOT={repo_root}',
        f'CONDA_ROOT={conda_root}',
        f'ENV_NAME={env_name}',
        f'RUN_ROOT={run_root}',
        f'RUN_LOCK_DIR={run_lock_dir}',
        'LOCK_ACQUIRED="0"',
        "",
        "cleanup_run_lock() {",
        '  if [[ "${LOCK_ACQUIRED}" == "1" ]]; then',
        '    rm -rf "${RUN_LOCK_DIR}"',
        "  fi",
        "}",
        "trap cleanup_run_lock EXIT",
        'if ! mkdir "${RUN_LOCK_DIR}" 2>/dev/null; then',
        '  echo "Another Runpod continuation launcher is already using ${RUN_ROOT}." >&2',
        "  exit 1",
        "fi",
        'LOCK_ACQUIRED="1"',
        'printf "%s\\n" "$$" > "${RUN_LOCK_DIR}/pid"',
        "",
        "install_system_deps() {",
        "  export DEBIAN_FRONTEND=noninteractive",
        "  apt-get update",
        "  apt-get install -y --no-install-recommends \\",
        "    build-essential \\",
        "    gfortran \\",
        "    git \\",
        "    libboost-all-dev \\",
        "    libfftw3-dev \\",
        "    libhdf5-dev \\",
        "    libhdf5-serial-dev \\",
        "    liblapack-dev \\",
        "    libnetcdf-dev \\",
        "    libnetcdff-dev \\",
        "    libopenblas-dev \\",
        "    libopenmpi-dev \\",
        "    openmpi-bin \\",
        "    python3-venv",
        "  rm -rf /var/lib/apt/lists/*",
        "}",
        "",
        "ensure_required_cuda_toolkit() {",
        f'  local required_release="{REQUIRED_RUNPOD_CUDA_TOOLKIT_RELEASE}"',
        '  local keyring_deb="/tmp/cuda-keyring_1.1-1_all.deb"',
        '  if [[ ! -d "/usr/local/cuda-${required_release}" ]]; then',
        "    python - <<'PY'",
        "import urllib.request",
        "urllib.request.urlretrieve(",
        '    "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb",',
        '    "/tmp/cuda-keyring_1.1-1_all.deb",',
        ")",
        "PY",
        '    dpkg -i "${keyring_deb}"',
        "    apt-get update",
        '    apt-get install -y --no-install-recommends "cuda-toolkit-${required_release//./-}"',
        '    rm -f "${keyring_deb}"',
        "  fi",
        '  ln -sfn "/usr/local/cuda-${required_release}" /usr/local/cuda',
        "}",
        "",
        'if [[ "${SYSTEM_DEPS_MODE}" == "always" ]]; then',
        "  install_system_deps",
        'elif [[ "${SYSTEM_DEPS_MODE}" == "auto" ]]; then',
        '  if [[ ! -x "${CONDA_ROOT}/bin/conda" ]] || ! command -v gfortran >/dev/null 2>&1; then',
        "    install_system_deps",
        "  fi",
        'elif [[ "${SYSTEM_DEPS_MODE}" != "never" ]]; then',
        '  echo "Unsupported --system-deps-mode: ${SYSTEM_DEPS_MODE}" >&2',
        "  exit 1",
        "fi",
        "",
        'if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then',
        "  python - <<'PY'",
        "import urllib.request",
        "urllib.request.urlretrieve(",
        '    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh",',
        '    "/tmp/miniforge.sh",',
        ")",
        "PY",
        '  bash /tmp/miniforge.sh -b -p "${CONDA_ROOT}"',
        "fi",
        "",
        'CUDA_TOOLKIT_RELEASE=""',
        'if [[ -L "/usr/local/cuda" ]]; then',
        '  CUDA_TOOLKIT_RELEASE="$(readlink -f /usr/local/cuda | sed -n \'s#.*/cuda-\\([0-9][0-9]*\\.[0-9][0-9]*\\).*#\\1#p\')"',
        "fi",
        'if [[ -z "${CUDA_TOOLKIT_RELEASE}" ]] && [[ -x "/usr/local/cuda/bin/nvcc" ]]; then',
        '  CUDA_TOOLKIT_RELEASE="$(/usr/local/cuda/bin/nvcc --version | sed -n \'s/.*release \\([0-9][0-9]*\\.[0-9][0-9]*\\).*/\\1/p\' | head -n 1)"',
        "fi",
        f'if [[ -z "${{CUDA_TOOLKIT_RELEASE}}" ]] || dpkg --compare-versions "${{CUDA_TOOLKIT_RELEASE}}" lt "{REQUIRED_RUNPOD_CUDA_TOOLKIT_RELEASE}"; then',
        "  ensure_required_cuda_toolkit",
        f'  CUDA_TOOLKIT_RELEASE="{REQUIRED_RUNPOD_CUDA_TOOLKIT_RELEASE}"',
        "fi",
        "",
        (
            "export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT="
            f"{shlex.quote(plan.pretend_version)}"
        ),
        'source "${CONDA_ROOT}/etc/profile.d/conda.sh"',
        'ENV_SPEC_PATH="${REPO_ROOT}/envs/jax.yml"',
        'ENV_HASH_PATH="${CONDA_ROOT}/envs/${ENV_NAME}/.simsopt-jax-env.sha256"',
        'ENV_SPEC_HASH="$(sha256sum "${ENV_SPEC_PATH}" | awk \'{print $1}\')"',
        'if conda env list | awk \'{print $1}\' | grep -Fxq "${ENV_NAME}"; then',
        '  if [[ -f "${ENV_HASH_PATH}" ]] && [[ "$(cat "${ENV_HASH_PATH}")" == "${ENV_SPEC_HASH}" ]]; then',
        '    echo "Conda env ${ENV_NAME} already matches ${ENV_SPEC_PATH}; skipping env update."',
        "  else",
        '    conda env update -n "${ENV_NAME}" -f "${ENV_SPEC_PATH}" --prune',
        '    printf "%s\\n" "${ENV_SPEC_HASH}" > "${ENV_HASH_PATH}"',
        "  fi",
        "else",
        '  conda env create -f "${ENV_SPEC_PATH}"',
        '  printf "%s\\n" "${ENV_SPEC_HASH}" > "${ENV_HASH_PATH}"',
        "fi",
        'conda activate "${ENV_NAME}"',
        'cd "${REPO_ROOT}"',
        'export SIMSOPT_JAX_CUDA_LIBRARY_MODE="bundled"',
        'python -m pip install --upgrade pip setuptools wheel',
        "if python - <<'PY'",
        "import importlib.metadata as metadata",
        'expected = "0.9.2"',
        "raise SystemExit(",
        "    0",
        "    if metadata.version('jax') == expected",
        "    and metadata.version('jaxlib') == expected",
        "    else 1",
        ")",
        "PY",
        "then",
        '  echo "jax/jaxlib 0.9.2 already installed; skipping force reinstall."',
        "else",
        f'  python -m pip install --upgrade --force-reinstall {shlex.quote(EXACT_JAX_GPU_WHEEL_SPEC)}',
        "fi",
        'python -m pip install -e ".[JAX_GPU,dev]"',
        "python - <<'PY'",
        "import jax",
        "import jaxlib",
        'expected = "0.9.2"',
        "if jax.__version__ != expected or jaxlib.__version__ != expected:",
        "    raise SystemExit(",
        '        f\"Expected jax/jaxlib {expected}, got jax={jax.__version__} jaxlib={jaxlib.__version__}\"',
        "    )",
        "PY",
        f"mkdir -p {shlex.quote(plan.remote_output_root)} {shlex.quote(plan.remote_cache_dir)}",
        "export PYTHONUNBUFFERED=1",
        "export HF_HUB_DISABLE_TELEMETRY=1",
        f"export SIMSOPT_BACKEND_MODE={shlex.quote(plan.backend_mode)}",
        'export SIMSOPT_BACKEND_STRICT=1',
        f"export SIMSOPT_JAX_TRANSFER_GUARD={shlex.quote(plan.transfer_guard)}",
        "export JAX_PLATFORMS=cuda",
        "export SIMSOPT_JAX_PLATFORM=cuda",
        "# Keep JAX from reserving most VRAM before continuation kernels allocate.",
        "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
        'export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true --xla_gpu_cuda_data_dir=/usr/local/cuda --xla_gpu_enable_llvm_module_compilation_parallelism=false"',
        f"export JAX_COMPILATION_CACHE_DIR={shlex.quote(plan.remote_cache_dir)}",
        "export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0",
        "export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1",
        "export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=all",
    ]
    command = [
        "python",
        "examples/single_stage_optimization/SINGLE_STAGE/run_single_stage_continuation.py",
        "--plasma-surf-filename",
        plan.plasma_surf_filename,
        "--equilibria-dir",
        _posix(PurePosixPath(plan.remote_equilibrium_path).parent),
        "--backend",
        "jax",
        "--optimizer-backend",
        "ondevice",
        "--trial-policy",
        plan.trial_policy,
        "--mpol",
        str(plan.mpol),
        "--ntor",
        str(plan.ntor),
        "--nphi",
        str(plan.nphi),
        "--ntheta",
        str(plan.ntheta),
        "--maxiter",
        str(plan.maxiter),
        "--coarse-maxiter",
        str(plan.coarse_maxiter),
        "--medium-maxiter",
        str(plan.medium_maxiter),
        "--prefinal-maxiter",
        str(plan.prefinal_maxiter),
        "--strict-validation",
    ]
    donor_dirs = remote_donor_run_dirs(plan)
    if len(donor_dirs) > 1:
        if plan.run_mode != "new":
            raise ValueError(
                "Runpod multi-donor continuation campaign only supports new runs."
            )
        command[2:2] = [
            "--output-root",
            plan.remote_output_root,
            "--run-id",
            plan.run_id,
        ]
        for donor_run_dir in donor_dirs:
            command.extend(["--campaign-donor-run-dir", donor_run_dir])
    elif plan.run_mode == "new":
        command[2:2] = [
            "--output-root",
            plan.remote_output_root,
            "--run-id",
            plan.run_id,
            "--initial-warm-start-run-dir",
            donor_dirs[0],
        ]
    elif plan.run_mode == "resume":
        command[2:2] = [
            "--resume-run-root",
            plan.remote_run_root,
            "--initial-warm-start-run-dir",
            donor_dirs[0],
        ]
    elif plan.run_mode == "summarize":
        command[2:2] = [
            "--summarize-run-root",
            plan.remote_run_root,
        ]
    else:
        raise ValueError(f"Unsupported Runpod continuation mode: {plan.run_mode}")
    if plan.max_final_field_error is not None:
        command.extend(
            ["--max-final-field-error", str(plan.max_final_field_error)]
        )
    if plan.max_final_abs_iota_error is not None:
        command.extend(
            ["--max-final-abs-iota-error", str(plan.max_final_abs_iota_error)]
        )
    if plan.max_final_non_qs is not None:
        command.extend(
            ["--max-final-non-qs", str(plan.max_final_non_qs)]
        )
    if plan.remote_jax_profile_dir is not None and plan.run_mode != "summarize":
        command.extend(
            ["--jax-profile-dir", plan.remote_jax_profile_dir]
        )
    command.extend(plan.single_stage_passthrough_args)
    lines.append(_shell_join(command))
    return "\n".join(lines)


def run_checked(command: list[str], *, input_text: str | None = None) -> None:
    subprocess.run(command, check=True, text=True, input=input_text)


def stream_repo_archive(
    *,
    plan: LaunchPlan,
    ssh_info: SshInfo,
) -> None:
    repo_relative_path = _relative_to_root(
        plan.local_repo_root,
        plan.local_columbia_root,
        description="local repo root",
    )
    remote_archive_path = remote_repo_archive_path(plan)
    with tempfile.TemporaryDirectory(prefix="runpod-repo-archive-") as tmpdir:
        local_archive_path = Path(tmpdir) / f"{repo_relative_path.name}.tar.gz"
        tar_command = build_repo_tar_command(
            local_columbia_root=plan.local_columbia_root,
            repo_relative_path=repo_relative_path,
            output_path=local_archive_path,
        )
        subprocess.run(
            tar_command,
            check=True,
            env=build_repo_tar_env(),
        )
        scp_to_remote(
            ssh_info=ssh_info,
            source=local_archive_path,
            remote_destination=remote_archive_path,
            recursive=False,
        )
        run_checked(
            build_remote_repo_extract_command(
                plan,
                ssh_info,
                remote_archive_path=remote_archive_path,
            )
        )


def run_ssh_script(
    *,
    ssh_info: SshInfo,
    script: str,
) -> None:
    command = [
        *ssh_base_command(ssh_info),
        ssh_info.target,
        "bash",
        "-s",
        "--",
    ]
    run_checked(command, input_text=script)


def scp_to_remote(
    *,
    ssh_info: SshInfo,
    source: Path,
    remote_destination: PurePosixPath,
    recursive: bool,
) -> None:
    command = scp_base_command(ssh_info)
    if recursive:
        command.append("-r")
    command.extend([str(source), f"{ssh_info.target}:{_posix(remote_destination)}"])
    run_checked(command)


def scp_from_remote(
    *,
    ssh_info: SshInfo,
    remote_source: PurePosixPath,
    local_destination: Path,
    recursive: bool,
) -> None:
    command = scp_base_command(ssh_info)
    if recursive:
        command.append("-r")
    command.extend([f"{ssh_info.target}:{_posix(remote_source)}", str(local_destination)])
    run_checked(command)


def resolve_launch_plan(
    args: argparse.Namespace,
    *,
    passthrough_args: tuple[str, ...] = (),
) -> LaunchPlan:
    if args.resume_existing_run and args.summarize_existing_run:
        raise SystemExit(
            "Pass at most one of --resume-existing-run or --summarize-existing-run."
        )
    local_columbia_root = _require_existing_path(
        Path(args.local_columbia_root),
        description="local Columbia root",
    )
    local_repo_root = _require_existing_path(
        Path(args.local_repo_root),
        description="local repo root",
    )
    pretend_version = args.pretend_version or _derive_pretend_version(
        local_repo_root
    )
    local_equilibrium_path = _require_existing_path(
        Path(args.equilibrium_path),
        description="equilibrium file",
    )
    donor_run_dir_args = (
        [str(DEFAULT_LOCAL_DONOR_RUN_DIR)]
        if args.donor_run_dir is None
        else [args.donor_run_dir]
        if isinstance(args.donor_run_dir, str)
        else list(args.donor_run_dir)
    )
    local_donor_run_dirs_value = tuple(
        _require_donor_artifacts(Path(path_value))
        for path_value in donor_run_dir_args
    )
    if not local_donor_run_dirs_value:
        raise SystemExit("Pass at least one --donor-run-dir.")
    if len(local_donor_run_dirs_value) > 1 and (
        args.resume_existing_run or args.summarize_existing_run
    ):
        raise SystemExit(
            "Multi-donor Runpod continuation currently supports only new runs."
        )
    local_donor_run_dir = local_donor_run_dirs_value[0]
    run_id = args.run_id or _run_id_now()
    local_output_dir = (
        Path(args.local_output_root).expanduser().resolve() / run_id
    )
    remote_columbia_root = PurePosixPath(args.remote_columbia_root)
    remote_repo_root = map_local_path_to_remote(
        local_repo_root,
        local_columbia_root=local_columbia_root,
        remote_columbia_root=remote_columbia_root,
    )
    remote_equilibrium_path = map_local_path_to_remote(
        local_equilibrium_path,
        local_columbia_root=local_columbia_root,
        remote_columbia_root=remote_columbia_root,
    )
    remote_donor_run_dirs_value = tuple(
        map_local_path_to_remote(
            donor_run_dir,
            local_columbia_root=local_columbia_root,
            remote_columbia_root=remote_columbia_root,
        )
        for donor_run_dir in local_donor_run_dirs_value
    )
    remote_donor_run_dir = remote_donor_run_dirs_value[0]
    remote_output_root = PurePosixPath(args.remote_output_root)
    campaign_mode = len(local_donor_run_dirs_value) > 1
    remote_run_root = (
        remote_output_root / f"campaign-{run_id}"
        if campaign_mode
        else remote_output_root / f"continuation-{run_id}"
    )
    remote_jax_profile_dir = resolve_remote_jax_profile_dir(
        remote_run_root=remote_run_root,
        requested_profile_dir=args.jax_profile_dir,
    )
    run_mode = (
        "summarize"
        if args.summarize_existing_run
        else "resume"
        if args.resume_existing_run
        else "new"
    )
    return LaunchPlan(
        pod_id=args.pod_id,
        run_id=run_id,
        pretend_version=pretend_version,
        local_columbia_root=local_columbia_root,
        local_repo_root=local_repo_root,
        local_equilibrium_path=local_equilibrium_path,
        local_donor_run_dir=local_donor_run_dir,
        local_output_dir=local_output_dir,
        remote_columbia_root=_posix(remote_columbia_root),
        remote_repo_root=_posix(remote_repo_root),
        remote_equilibrium_path=_posix(remote_equilibrium_path),
        remote_donor_run_dir=_posix(remote_donor_run_dir),
        remote_output_root=_posix(remote_output_root),
        remote_run_root=_posix(remote_run_root),
        remote_jax_profile_dir=None
        if remote_jax_profile_dir is None
        else _posix(remote_jax_profile_dir),
        remote_summary_path=_posix(
            remote_run_root
            / ("campaign_summary.json" if campaign_mode else "continuation_summary.json")
        ),
        remote_validation_path=None
        if campaign_mode
        else _posix(remote_run_root / "continuation_validation.json"),
        remote_cache_dir=_posix(PurePosixPath(args.remote_cache_dir)),
        remote_conda_root=_posix(PurePosixPath(args.remote_conda_root)),
        plasma_surf_filename=args.plasma_surf_filename,
        system_deps_mode=args.system_deps_mode,
        backend_mode=args.backend_mode,
        transfer_guard=args.transfer_guard,
        trial_policy=args.trial_policy,
        run_mode=run_mode,
        max_final_field_error=args.max_final_field_error,
        max_final_abs_iota_error=args.max_final_abs_iota_error,
        max_final_non_qs=args.max_final_non_qs,
        mpol=int(args.mpol),
        ntor=int(args.ntor),
        nphi=int(args.nphi),
        ntheta=int(args.ntheta),
        maxiter=int(args.maxiter),
        coarse_maxiter=int(args.coarse_maxiter),
        medium_maxiter=int(args.medium_maxiter),
        prefinal_maxiter=int(args.prefinal_maxiter),
        fetch_full_run_root=bool(args.fetch_full_run_root),
        single_stage_passthrough_args=passthrough_args,
        local_donor_run_dirs=local_donor_run_dirs_value,
        remote_donor_run_dirs=tuple(
            _posix(remote_donor_run_dir_value)
            for remote_donor_run_dir_value in remote_donor_run_dirs_value
        ),
    )


def _write_plan_json(plan: LaunchPlan, ssh_info: SshInfo) -> None:
    plan.local_output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan": {
            **asdict(plan),
            "local_columbia_root": str(plan.local_columbia_root),
            "local_repo_root": str(plan.local_repo_root),
            "local_equilibrium_path": str(plan.local_equilibrium_path),
            "local_donor_run_dir": str(plan.local_donor_run_dir),
            "local_donor_run_dirs": [
                str(path) for path in local_donor_run_dirs(plan)
            ],
            "local_output_dir": str(plan.local_output_dir),
        },
        "ssh": {
            "host": ssh_info.host,
            "port": ssh_info.port,
            "key_path": str(ssh_info.key_path),
            "user": ssh_info.user,
            "pod_id": ssh_info.pod_id,
        },
    }
    with open(plan.local_output_dir / "launch_plan.json", "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2, sort_keys=True)


def print_dry_run(plan: LaunchPlan, ssh_info: SshInfo) -> None:
    repo_relative_path = _relative_to_root(
        plan.local_repo_root,
        plan.local_columbia_root,
        description="local repo root",
    )
    repo_sync = (
        f"{_shell_join(build_repo_tar_command(local_columbia_root=plan.local_columbia_root, repo_relative_path=repo_relative_path))} "
        f"| {_shell_join(build_remote_repo_extract_command(plan, ssh_info, remote_archive_path=remote_repo_archive_path(plan)))}"
    )
    equilibrium_copy = _shell_join(
        [
            *scp_base_command(ssh_info),
            str(plan.local_equilibrium_path),
            f"{ssh_info.target}:{_posix(PurePosixPath(plan.remote_equilibrium_path).parent)}",
        ]
    )
    donor_copy = [
        _shell_join(
            [
                *scp_base_command(ssh_info),
                "-r",
                str(local_donor_run_dir_value),
                f"{ssh_info.target}:{_posix(PurePosixPath(remote_donor_run_dir_value).parent)}",
            ]
        )
        for local_donor_run_dir_value, remote_donor_run_dir_value in zip(
            local_donor_run_dirs(plan),
            remote_donor_run_dirs(plan),
        )
    ]
    print(
        json.dumps(
            {
                "plan": {
                    **asdict(plan),
                    "local_columbia_root": str(plan.local_columbia_root),
                    "local_repo_root": str(plan.local_repo_root),
                    "local_equilibrium_path": str(plan.local_equilibrium_path),
                    "local_donor_run_dir": str(plan.local_donor_run_dir),
                    "local_donor_run_dirs": [
                        str(path) for path in local_donor_run_dirs(plan)
                    ],
                    "local_output_dir": str(plan.local_output_dir),
                },
                "ssh": {
                    "host": ssh_info.host,
                    "port": ssh_info.port,
                    "key_path": str(ssh_info.key_path),
                    "user": ssh_info.user,
                    "pod_id": ssh_info.pod_id,
                },
                "commands": {
                    "repo_sync": repo_sync,
                    "remote_prepare_script": build_remote_prepare_script(plan),
                    "equilibrium_copy": equilibrium_copy,
                    "donor_copy": donor_copy,
                    "remote_execution_script": build_remote_execution_script(plan),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, tuple[str, ...]]:
    parser = argparse.ArgumentParser(
        description=(
            "Sync the current local simsopt-jax workspace plus donor artifacts to a "
            "Runpod pod, bootstrap the GPU environment, run the single-stage "
            "continuation workflow, and fetch the validation reports. Extra args "
            "after '--' are forwarded to the remote continuation driver."
        )
    )
    parser.add_argument("--pod-id", default=DEFAULT_POD_ID)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--ssh-key", default=None)
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument(
        "--local-columbia-root",
        default=str(LOCAL_COLUMBIA_ROOT),
    )
    parser.add_argument(
        "--local-repo-root",
        default=str(REPO_ROOT),
    )
    parser.add_argument(
        "--equilibrium-path",
        default=str(DEFAULT_LOCAL_EQUILIBRIUM_PATH),
    )
    parser.add_argument(
        "--donor-run-dir",
        action="append",
        default=None,
        help=(
            "Local donor run directory to sync and use as a continuation donor. "
            "May be passed multiple times to run a multi-donor campaign."
        ),
    )
    parser.add_argument(
        "--plasma-surf-filename",
        default=DEFAULT_PLASMA_SURF_FILENAME,
    )
    parser.add_argument(
        "--remote-columbia-root",
        default=_posix(DEFAULT_REMOTE_COLUMBIA_ROOT),
    )
    parser.add_argument(
        "--remote-output-root",
        default=_posix(DEFAULT_REMOTE_OUTPUT_ROOT),
    )
    parser.add_argument(
        "--remote-cache-dir",
        default=_posix(DEFAULT_REMOTE_CACHE_DIR),
    )
    parser.add_argument(
        "--jax-profile-dir",
        default=None,
        help=(
            "Optional remote JAX/XProf trace root threaded into the staged "
            "continuation run. Relative paths are resolved under the remote "
            "continuation run root."
        ),
    )
    parser.add_argument(
        "--remote-conda-root",
        default=_posix(DEFAULT_REMOTE_CONDA_ROOT),
    )
    parser.add_argument(
        "--system-deps-mode",
        choices=("auto", "always", "never"),
        default=DEFAULT_SYSTEM_DEPS_MODE,
    )
    parser.add_argument(
        "--backend-mode",
        default=DEFAULT_BACKEND_MODE,
    )
    parser.add_argument(
        "--transfer-guard",
        default=DEFAULT_TRANSFER_GUARD,
    )
    parser.add_argument(
        "--trial-policy",
        choices=("none", "validated-fast"),
        default="validated-fast",
    )
    parser.add_argument("--mpol", type=int, default=8)
    parser.add_argument("--ntor", type=int, default=6)
    parser.add_argument("--nphi", type=int, default=255)
    parser.add_argument("--ntheta", type=int, default=64)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--coarse-maxiter", type=int, default=1)
    parser.add_argument("--medium-maxiter", type=int, default=1)
    parser.add_argument("--prefinal-maxiter", type=int, default=2)
    parser.add_argument("--max-final-field-error", type=float, default=5e-4)
    parser.add_argument("--max-final-abs-iota-error", type=float, default=None)
    parser.add_argument("--max-final-non-qs", type=float, default=0.05)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Stable run identifier. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--resume-existing-run",
        action="store_true",
        help="Reuse completed stages under continuation-<run-id> and continue from the first incomplete stage.",
    )
    parser.add_argument(
        "--summarize-existing-run",
        action="store_true",
        help="Only reconstruct summary/validation artifacts for continuation-<run-id> on the remote pod.",
    )
    parser.add_argument(
        "--local-output-root",
        default=str(DEFAULT_LOCAL_OUTPUT_ROOT),
    )
    parser.add_argument(
        "--pretend-version",
        default=None,
        help=(
            "Override the setuptools-scm pretend version used for remote editable "
            "installs. Defaults to 0.0.dev0+g<local-short-head>."
        ),
    )
    parser.add_argument(
        "--fetch-full-run-root",
        action="store_true",
        help="Also copy the full remote continuation run root back locally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved Runpod plan and commands without executing them.",
    )
    args, passthrough_args = parser.parse_known_args(argv)
    return args, _normalize_passthrough_args(passthrough_args)


def fetch_results(plan: LaunchPlan, ssh_info: SshInfo) -> FetchReport:
    plan.local_output_dir.mkdir(parents=True, exist_ok=True)
    remote_profiling_report_path = resolve_remote_profiling_report_path(plan)
    artifacts = [
        FetchArtifact(
            label="summary",
            remote_source=PurePosixPath(plan.remote_summary_path),
            local_destination=plan.local_output_dir
            / PurePosixPath(plan.remote_summary_path).name,
            recursive=False,
            required=True,
        ),
        FetchArtifact(
            label="profiling_report",
            remote_source=remote_profiling_report_path,
            local_destination=plan.local_output_dir
            / remote_profiling_report_path.name,
            recursive=False,
            required=False,
        ),
    ]
    if plan.remote_validation_path is not None:
        artifacts.insert(
            1,
            FetchArtifact(
                label="validation",
                remote_source=PurePosixPath(plan.remote_validation_path),
                local_destination=plan.local_output_dir
                / PurePosixPath(plan.remote_validation_path).name,
                recursive=False,
                required=False,
            ),
        )
    if plan.fetch_full_run_root or launch_plan_uses_campaign(plan):
        artifacts.append(
            FetchArtifact(
                label="run_root",
                remote_source=PurePosixPath(plan.remote_run_root),
                local_destination=plan.local_output_dir,
                recursive=True,
                required=False,
            )
        )

    fetched_paths: list[str] = []
    required_failures: list[str] = []
    optional_failures: list[str] = []
    for artifact in artifacts:
        try:
            scp_from_remote(
                ssh_info=ssh_info,
                remote_source=artifact.remote_source,
                local_destination=artifact.local_destination,
                recursive=artifact.recursive,
            )
        except subprocess.CalledProcessError:
            failure = f"{artifact.label}:{_posix(artifact.remote_source)}"
            if artifact.required:
                required_failures.append(failure)
            else:
                optional_failures.append(failure)
        else:
            fetched_paths.append(str(artifact.local_destination))
    return FetchReport(
        fetched_paths=tuple(fetched_paths),
        required_failures=tuple(required_failures),
        optional_failures=tuple(optional_failures),
    )


def resolve_local_single_stage_scan_root(plan: LaunchPlan) -> Path:
    fetched_run_root = plan.local_output_dir / PurePosixPath(plan.remote_run_root).name
    if fetched_run_root.exists():
        return fetched_run_root
    return plan.local_output_dir


def resolve_local_stage2_scan_root(plan: LaunchPlan) -> Path:
    donor_parent_dirs = [path.parent for path in local_donor_run_dirs(plan)]
    common_root = os.path.commonpath([str(path) for path in donor_parent_dirs])
    return Path(common_root)


def build_candidate_ledger_command(plan: LaunchPlan) -> list[str]:
    command = [
        sys.executable,
        str(_CANDIDATE_LEDGER_SCRIPT),
        "--stage2-root",
        str(resolve_local_stage2_scan_root(plan)),
        "--single-stage-root",
        str(resolve_local_single_stage_scan_root(plan)),
        "--output-json",
        str(plan.local_output_dir / "candidate_ledger.json"),
    ]
    if plan.max_final_field_error is not None:
        command.extend(
            ["--single-stage-max-final-field-error", str(plan.max_final_field_error)]
        )
    if plan.max_final_abs_iota_error is not None:
        command.extend(
            [
                "--single-stage-max-final-abs-iota-error",
                str(plan.max_final_abs_iota_error),
            ]
        )
    if plan.max_final_non_qs is not None:
        command.extend(
            ["--single-stage-max-final-non-qs", str(plan.max_final_non_qs)]
        )
    return command


def build_local_candidate_ledger(plan: LaunchPlan) -> Path:
    command = build_candidate_ledger_command(plan)
    subprocess.run(
        command,
        check=True,
        cwd=str(REPO_ROOT),
    )
    return plan.local_output_dir / "candidate_ledger.json"


def main() -> None:
    args, passthrough_args = parse_args()
    plan = resolve_launch_plan(args, passthrough_args=passthrough_args)
    ssh_info = resolve_ssh_info(args)
    _write_plan_json(plan, ssh_info)
    if args.dry_run:
        print_dry_run(plan, ssh_info)
        return

    stream_repo_archive(plan=plan, ssh_info=ssh_info)
    run_ssh_script(ssh_info=ssh_info, script=build_remote_prepare_script(plan))
    scp_to_remote(
        ssh_info=ssh_info,
        source=plan.local_equilibrium_path,
        remote_destination=PurePosixPath(plan.remote_equilibrium_path).parent,
        recursive=False,
    )
    for local_donor_run_dir_value, remote_donor_run_dir_value in zip(
        local_donor_run_dirs(plan),
        remote_donor_run_dirs(plan),
    ):
        scp_to_remote(
            ssh_info=ssh_info,
            source=local_donor_run_dir_value,
            remote_destination=PurePosixPath(remote_donor_run_dir_value).parent,
            recursive=True,
        )

    remote_failed = False
    try:
        run_ssh_script(ssh_info=ssh_info, script=build_remote_execution_script(plan))
    except subprocess.CalledProcessError:
        remote_failed = True

    fetch_report = fetch_results(plan, ssh_info)
    fetch_failed = fetch_report.has_required_failures

    candidate_ledger_path: Path | None = None
    candidate_ledger_failed = False
    if not fetch_failed:
        try:
            candidate_ledger_path = build_local_candidate_ledger(plan)
        except subprocess.CalledProcessError:
            candidate_ledger_failed = True

    if fetch_failed:
        raise SystemExit(
            "Runpod continuation finished but required artifact fetch failed. "
            f"Missing: {', '.join(fetch_report.required_failures)}. "
            f"Partial artifacts, if any, are under {plan.local_output_dir}."
        )
    if candidate_ledger_failed:
        raise SystemExit(
            "Fetched continuation artifacts locally, but candidate ledger generation "
            f"failed under {plan.local_output_dir}."
        )
    if remote_failed:
        optional_suffix = (
            ""
            if not fetch_report.has_optional_failures
            else " Optional fetch failures: "
            + ", ".join(fetch_report.optional_failures)
            + "."
        )
        raise SystemExit(
            "Remote continuation run failed. "
            f"Fetched partial reports to {plan.local_output_dir} for inspection."
            + optional_suffix
        )
    print(
        json.dumps(
            {
                "local_output_dir": str(plan.local_output_dir),
                "summary_path": str(
                    plan.local_output_dir / PurePosixPath(plan.remote_summary_path).name
                ),
                "validation_path": None
                if plan.remote_validation_path is None
                else str(
                    plan.local_output_dir
                    / PurePosixPath(plan.remote_validation_path).name
                ),
                "optional_fetch_failures": list(fetch_report.optional_failures),
                "candidate_ledger_path": None
                if candidate_ledger_path is None
                else str(candidate_ledger_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
