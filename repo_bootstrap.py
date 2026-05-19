from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import os
import sys
import types
from pathlib import Path

_BOOTSTRAP_VERSION = "0.0.dev0+source"
_BOOTSTRAP_VERSION_TUPLE = (0, 0, "dev0", "source")
_ENTRYPOINT_PLATFORM_CHOICES = ("auto", "cpu", "cuda", "mps")
_ENTRYPOINT_PLATFORM_ENV_VARS = (
    "JAX_PLATFORMS",
    "SIMSOPT_JAX_PLATFORM",
    "SIMSOPT_JAX_BACKEND",
)
_JAX_PLATFORMS_ENV = "JAX_PLATFORMS"
_JAX_ENABLE_X64_ENV = "JAX_ENABLE_X64"
_XLA_PREALLOCATE_ENV = "XLA_PYTHON_CLIENT_PREALLOCATE"
_XLA_FLAGS_ENV = "XLA_FLAGS"
_XLA_GPU_CUDA_DATA_DIR_FLAG = "--xla_gpu_cuda_data_dir="
_CUDA_TOOLCHAIN_ROOT_ENV = "SIMSOPT_CUDA_TOOLCHAIN_ROOT"
_CUDA_LIBRARY_MODE_ENV = "SIMSOPT_JAX_CUDA_LIBRARY_MODE"
_CUDA_LIBRARY_MODE_AUTO = "auto"
_CUDA_LIBRARY_MODE_LOCAL = "local"
_CUDA_LIBRARY_MODE_BUNDLED = "bundled"
_CUDA_LIBRARY_MODES = (
    _CUDA_LIBRARY_MODE_AUTO,
    _CUDA_LIBRARY_MODE_LOCAL,
    _CUDA_LIBRARY_MODE_BUNDLED,
)
_DEFAULT_CUDA_TOOLCHAIN_ROOT = Path("/usr/local/cuda")
_LD_LIBRARY_PATH_ENV = "LD_LIBRARY_PATH"


def _prepend_env_path(env: dict[str, str], name: str, entry: Path) -> None:
    """Prepend one path entry without duplicating it."""
    entry_str = str(entry)
    current = env.get(name)
    if current is None or current == "":
        env[name] = entry_str
        return
    parts = current.split(os.pathsep)
    if parts and parts[0] == entry_str:
        return
    env[name] = os.pathsep.join([entry_str, *parts])


def _normalize_cuda_library_mode(value: str | None) -> str:
    """Normalize the CUDA runtime-library mode selector."""
    if value is None:
        return _CUDA_LIBRARY_MODE_AUTO
    normalized = value.strip().lower()
    if normalized == "":
        return _CUDA_LIBRARY_MODE_AUTO
    if normalized not in _CUDA_LIBRARY_MODES:
        raise ValueError(
            f"Unsupported CUDA library mode {value!r}; expected one of "
            f"{_CUDA_LIBRARY_MODES}."
        )
    return normalized


def _drop_xla_flags_with_prefix(env: dict[str, str], prefix: str) -> None:
    """Remove XLA flag tokens with ``prefix`` while preserving unrelated flags."""
    current = env.get(_XLA_FLAGS_ENV)
    if current is None or current == "":
        return
    tokens = [token for token in current.split() if not token.startswith(prefix)]
    if tokens:
        env[_XLA_FLAGS_ENV] = " ".join(tokens)
        return
    env.pop(_XLA_FLAGS_ENV, None)


def _has_cuda_toolchain_binaries(root: Path) -> bool:
    """Return whether a root exposes CUDA linker/compiler tools."""
    return any((root / "bin" / tool).exists() for tool in ("nvlink", "ptxas"))


def _candidate_env_cuda_toolchain_roots(env: dict[str, str]) -> list[Path]:
    """Return active Python/virtual environment roots that may carry CUDA tools."""
    candidates: list[Path] = []
    for env_name in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        env_value = env.get(env_name)
        if env_value:
            candidates.append(Path(env_value).expanduser())
    candidates.append(Path(sys.prefix).expanduser())
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _resolve_cuda_toolchain_root(env: dict[str, str]) -> Path | None:
    """Return a usable CUDA toolkit root for external compiler tools."""
    explicit_root = env.get(_CUDA_TOOLCHAIN_ROOT_ENV)
    candidates: list[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root).expanduser())
    candidates.extend(_candidate_env_cuda_toolchain_roots(env))
    candidates.append(_DEFAULT_CUDA_TOOLCHAIN_ROOT)
    for candidate in candidates:
        if not (candidate / "bin").is_dir():
            continue
        if candidate == _DEFAULT_CUDA_TOOLCHAIN_ROOT or _has_cuda_toolchain_binaries(
            candidate
        ):
            return candidate
    return None


def _resolve_cuda_nvjitlink_lib_root(cuda_root: Path) -> Path | None:
    """Return an nvJitLink library directory that should shadow older system CUDA."""
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    target_lib_roots = ()
    targets_root = cuda_root / "targets"
    if targets_root.is_dir():
        target_lib_roots = tuple(
            sorted(path for path in targets_root.glob("*/lib") if path.is_dir())
        )
    candidates = (
        cuda_root / "lib64",
        *target_lib_roots,
        cuda_root
        / "lib"
        / python_version
        / "site-packages"
        / "nvidia"
        / "nvjitlink"
        / "lib",
    )
    for candidate in candidates:
        if (candidate / "libnvJitLink.so.12").exists():
            return candidate
    return None


def apply_cuda_toolchain_env(env: dict[str, str]) -> None:
    """Point JAX/XLA at a concrete external CUDA toolkit when one exists."""
    cuda_library_mode = _normalize_cuda_library_mode(env.get(_CUDA_LIBRARY_MODE_ENV))
    if cuda_library_mode == _CUDA_LIBRARY_MODE_BUNDLED:
        # JAX's pip-installed CUDA wheels carry their own CUDA user-space
        # libraries; forcing a local toolkit path here can make subprocesses
        # resolve the wrong runtime libraries.
        env.pop(_LD_LIBRARY_PATH_ENV, None)
        _drop_xla_flags_with_prefix(env, _XLA_GPU_CUDA_DATA_DIR_FLAG)
        return
    cuda_root = _resolve_cuda_toolchain_root(env)
    if cuda_root is None:
        return
    _prepend_env_path(env, "PATH", cuda_root / "bin")
    nvjitlink_lib_root = _resolve_cuda_nvjitlink_lib_root(cuda_root)
    if nvjitlink_lib_root is not None:
        _prepend_env_path(env, _LD_LIBRARY_PATH_ENV, nvjitlink_lib_root)
    existing_xla_flags = env.get(_XLA_FLAGS_ENV)
    if existing_xla_flags and _XLA_GPU_CUDA_DATA_DIR_FLAG in existing_xla_flags:
        return
    data_dir_flag = f"{_XLA_GPU_CUDA_DATA_DIR_FLAG}{cuda_root}"
    if existing_xla_flags:
        env[_XLA_FLAGS_ENV] = f"{data_dir_flag} {existing_xla_flags}"
        return
    env[_XLA_FLAGS_ENV] = data_dir_flag


def _normalize_entrypoint_platform_token(platform: str | None) -> str | None:
    """Normalize one benchmark/example platform token."""
    if platform is None:
        return None
    normalized = str(platform).strip().lower()
    if normalized == "":
        return None
    if normalized == "gpu":
        normalized = "cuda"
    if normalized == "auto":
        return None
    if normalized not in _ENTRYPOINT_PLATFORM_CHOICES[1:]:
        raise ValueError(
            f"Unsupported JAX platform {platform!r}; expected one of "
            f"{_ENTRYPOINT_PLATFORM_CHOICES}."
        )
    return normalized


def _normalize_entrypoint_platform_spec(
    platform: str | None,
    *,
    allow_multiple: bool,
) -> str | None:
    """Normalize an entrypoint platform selector or JAX platform list."""
    if platform is None:
        return None
    normalized = str(platform).strip().lower()
    if normalized == "":
        return None
    if not allow_multiple or "," not in normalized:
        return _normalize_entrypoint_platform_token(normalized)
    parts = [
        normalized_part
        for normalized_part in (
            _normalize_entrypoint_platform_token(part)
            for part in normalized.split(",")
        )
        if normalized_part is not None
    ]
    if not parts:
        return None
    return ",".join(parts)


def _argv_requests_flag(argv: list[str], flags: tuple[str, ...]) -> bool:
    if not flags:
        return False
    flag_set = set(flags)
    return any(arg in flag_set for arg in argv)


def with_cpu_callback_lane(platform: str | None) -> str | None:
    """Return the JAX_PLATFORMS spec with CPU appended for callback fallback.

    JAX requires CPU to be a registered platform when ``io_callback`` /
    ``pure_callback`` / ``debug.callback`` execute on the CUDA lane. This is
    the single source of truth for "CUDA needs CPU as fallback". The CPU is
    inserted immediately after CUDA so JAX still picks CUDA as the default
    backend.
    """
    normalized = _normalize_entrypoint_platform_spec(platform, allow_multiple=True)
    if normalized is None:
        return None
    parts = normalized.split(",")
    if ("cuda" not in parts) or ("cpu" in parts):
        return normalized
    return ",".join(["cuda", "cpu", *[part for part in parts if part != "cuda"]])


def preparse_entrypoint_jax_platform(
    argv: list[str],
    *,
    default: str | None = None,
    respect_existing_env: bool = True,
) -> str | None:
    """Resolve an early JAX platform request without importing ``jax``."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--platform", choices=_ENTRYPOINT_PLATFORM_CHOICES, default=None
    )
    args, _ = parser.parse_known_args(argv)
    if args.platform is not None:
        return _normalize_entrypoint_platform_spec(
            args.platform, allow_multiple=False
        )
    if respect_existing_env:
        for name in _ENTRYPOINT_PLATFORM_ENV_VARS:
            env_value = _normalize_entrypoint_platform_spec(
                os.environ.get(name),
                allow_multiple=name == _JAX_PLATFORMS_ENV,
            )
            if env_value is not None:
                return env_value
    return _normalize_entrypoint_platform_spec(default, allow_multiple=False)


def apply_entrypoint_jax_runtime_env(platform: str | None) -> str | None:
    """Synchronize platform/x64 env vars before any entrypoint imports ``jax``."""
    normalized = _normalize_entrypoint_platform_spec(platform, allow_multiple=True)
    normalized = with_cpu_callback_lane(normalized)
    os.environ.setdefault(_JAX_ENABLE_X64_ENV, "True")
    for name in _ENTRYPOINT_PLATFORM_ENV_VARS:
        os.environ.pop(name, None)
    os.environ.pop(_XLA_PREALLOCATE_ENV, None)
    if normalized is None:
        return None
    normalized_parts = normalized.split(",")
    default_platform = normalized_parts[0]
    os.environ[_JAX_PLATFORMS_ENV] = normalized
    for name in _ENTRYPOINT_PLATFORM_ENV_VARS:
        if name == _JAX_PLATFORMS_ENV:
            continue
        os.environ[name] = default_platform
    if "cuda" in normalized_parts:
        os.environ.setdefault(_XLA_PREALLOCATE_ENV, "false")
        apply_cuda_toolchain_env(os.environ)
    return default_platform


def configure_entrypoint_jax_runtime(
    argv: list[str] | None = None,
    *,
    default_platform: str | None = None,
    respect_existing_env: bool = True,
    require_cpu_platform_when_flags: tuple[str, ...] = (),
) -> str | None:
    """Apply entrypoint JAX env policy before any direct ``import jax``."""
    resolved_argv = [] if argv is None else argv
    requested_platform = preparse_entrypoint_jax_platform(
        resolved_argv,
        default=default_platform,
        respect_existing_env=respect_existing_env,
    )
    if _argv_requests_flag(resolved_argv, require_cpu_platform_when_flags):
        requested_platform = with_cpu_callback_lane(requested_platform)
    return apply_entrypoint_jax_runtime_env(requested_platform)


def _install_bootstrap_version_stub(package_root: Path) -> None:
    """Provide ``simsopt._version`` when bootstrapping a clean source tree."""
    version_file = package_root / "_version.py"
    if version_file.exists():
        return

    module = types.ModuleType("simsopt._version")
    module.version = _BOOTSTRAP_VERSION
    module.__version__ = _BOOTSTRAP_VERSION
    module.version_tuple = _BOOTSTRAP_VERSION_TUPLE
    module.__version_tuple__ = _BOOTSTRAP_VERSION_TUPLE
    module.__all__ = (
        "__version__",
        "__version_tuple__",
        "version",
        "version_tuple",
    )
    sys.modules["simsopt._version"] = module


def _is_simsopt_editable_finder(finder: object) -> bool:
    """Return True when ``finder`` is the local simsopt editable redirector."""
    finder_module = type(finder).__module__
    if finder_module == "_simsopt_editable":
        return True
    if not finder_module.startswith("__editable__"):
        return False
    return "simsopt" in finder_module.lower()


def _strip_editable_finders() -> None:
    """Remove editable-import finders that can hijack local ``simsopt`` imports."""
    sys.meta_path = [
        finder for finder in sys.meta_path if not _is_simsopt_editable_finder(finder)
    ]


def _is_same_local_package(module: types.ModuleType | None, package_root: Path) -> bool:
    """Return True when ``module`` already resolves to this source tree."""
    if module is None:
        return False

    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        try:
            if Path(module_file).resolve() == (package_root / "__init__.py").resolve():
                return True
        except OSError:
            return False

    module_path = getattr(module, "__path__", None)
    if module_path:
        try:
            return any(
                Path(path).resolve() == package_root.resolve() for path in module_path
            )
        except OSError:
            return False

    return False


def _is_same_local_extension(
    module: types.ModuleType | None, extension_path: Path
) -> bool:
    """Return True when ``module`` already resolves to ``extension_path``."""
    if module is None:
        return False
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False
    try:
        return Path(module_file).resolve() == extension_path.resolve()
    except OSError:
        return False


def _find_local_simsoptpp_extension(repo_root: Path) -> Path | None:
    """Return the first repo-local ``simsoptpp`` extension matching this Python."""
    build_root = repo_root / "build"
    if not build_root.is_dir():
        return None

    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        for candidate in sorted(build_root.glob(f"**/simsoptpp{suffix}")):
            try:
                return candidate.resolve()
            except OSError:
                continue
    return None


def _load_extension_module(module_name: str, extension_path: Path) -> types.ModuleType:
    """Load a compiled extension directly from ``extension_path``."""
    resolved_extension_path = extension_path.resolve()
    spec = importlib.machinery.PathFinder.find_spec(
        module_name, [str(extension_path.parent)]
    )
    if spec is None or spec.loader is None or spec.origin is None:
        raise RuntimeError(
            f"Failed to resolve {module_name} from local extension {resolved_extension_path}"
        )
    if Path(spec.origin).resolve() != resolved_extension_path:
        raise RuntimeError(
            f"Resolved {module_name} from unexpected origin {spec.origin}, "
            f"expected {resolved_extension_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _bootstrap_local_simsoptpp(repo_root: Path) -> bool:
    """Load the repo-local ``simsoptpp`` extension when one has been built."""
    extension_path = _find_local_simsoptpp_extension(repo_root)
    if extension_path is None:
        return False

    existing = sys.modules.get("simsoptpp")
    if _is_same_local_extension(existing, extension_path):
        return False

    sys.modules.pop("simsoptpp", None)
    _load_extension_module("simsoptpp", extension_path)
    return True


def _module_resolves_within_package(
    module: types.ModuleType | None,
    package_root: Path,
) -> bool:
    """Return True when ``module`` resolves somewhere inside ``package_root``."""
    if module is None:
        return False

    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        try:
            return Path(module_file).resolve().is_relative_to(package_root.resolve())
        except OSError:
            return False

    module_path = getattr(module, "__path__", None)
    if module_path:
        try:
            return any(
                Path(path).resolve().is_relative_to(package_root.resolve())
                for path in module_path
            )
        except OSError:
            return False

    return False


def _is_bootstrap_version_stub(module: types.ModuleType) -> bool:
    """Return True when ``module`` is the synthetic ``simsopt._version`` stub."""
    return (
        getattr(module, "version", None) == _BOOTSTRAP_VERSION
        and getattr(module, "__version__", None) == _BOOTSTRAP_VERSION
        and getattr(module, "version_tuple", None) == _BOOTSTRAP_VERSION_TUPLE
        and getattr(module, "__version_tuple__", None) == _BOOTSTRAP_VERSION_TUPLE
    )


def _find_detached_simsopt_submodules(package_root: Path) -> set[str]:
    """Return loaded ``simsopt.*`` modules detached from the canonical package chain."""
    canonical_names: set[str] = set()
    detached_names: set[str] = set()
    module_names = sorted(
        (
            name
            for name in sys.modules
            if name == "simsopt" or name.startswith("simsopt.")
        ),
        key=lambda name: (name.count("."), name),
    )

    for name in module_names:
        module = sys.modules.get(name)
        if module is None:
            continue

        if name == "simsopt":
            is_canonical = _is_same_local_package(module, package_root)
        elif name == "simsopt._version":
            is_canonical = "simsopt" in canonical_names and (
                _module_resolves_within_package(module, package_root)
                or _is_bootstrap_version_stub(module)
            )
        else:
            parent_name, child_name = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            is_canonical = (
                parent_name in canonical_names
                and _module_resolves_within_package(module, package_root)
                and parent is not None
                and getattr(parent, child_name, None) is module
            )

        if is_canonical:
            canonical_names.add(name)
        else:
            detached_names.add(name)

    return detached_names


def _purge_modules(module_names: set[str]) -> None:
    """Drop modules from ``sys.modules`` and unlink matching parent attributes."""
    for name in sorted(
        module_names,
        key=lambda module_name: module_name.count("."),
        reverse=True,
    ):
        module = sys.modules.pop(name, None)
        if module is None or "." not in name:
            continue

        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None and getattr(parent, child_name, None) is module:
            delattr(parent, child_name)


def bootstrap_local_simsopt(src_root: str | Path) -> None:
    """Force imports to resolve against this repo's local simsopt source tree."""
    src_root_path = Path(src_root)
    package_root = src_root_path / "simsopt"
    repo_root = src_root_path.parent
    _strip_editable_finders()
    simsoptpp_reloaded = _bootstrap_local_simsoptpp(repo_root)
    existing = sys.modules.get("simsopt")

    if _is_same_local_package(existing, package_root) and not simsoptpp_reloaded:
        _install_bootstrap_version_stub(package_root)
        _purge_modules(_find_detached_simsopt_submodules(package_root))
        return

    _purge_modules(
        {
            name
            for name in list(sys.modules)
            if name == "simsopt" or name.startswith("simsopt.")
        }
    )
    _install_bootstrap_version_stub(package_root)
    spec = importlib.util.spec_from_file_location(
        "simsopt",
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Failed to bootstrap local simsopt package from {package_root}"
        )
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(package_root)]
    sys.modules["simsopt"] = module
    spec.loader.exec_module(module)
