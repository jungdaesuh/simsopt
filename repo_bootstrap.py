from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

_BOOTSTRAP_VERSION = "0.0.dev0+source"
_BOOTSTRAP_VERSION_TUPLE = (0, 0, "dev0", "source")
_ENTRYPOINT_PLATFORM_CHOICES = ("auto", "cpu", "cuda")
_ENTRYPOINT_PLATFORM_ENV_VARS = (
    "JAX_PLATFORMS",
    "SIMSOPT_JAX_PLATFORM",
    "SIMSOPT_JAX_BACKEND",
)
_JAX_ENABLE_X64_ENV = "JAX_ENABLE_X64"
_XLA_PREALLOCATE_ENV = "XLA_PYTHON_CLIENT_PREALLOCATE"


def _normalize_entrypoint_platform(platform: str | None) -> str | None:
    """Normalize benchmark/example platform tokens to the JAX selector env contract."""
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
        return _normalize_entrypoint_platform(args.platform)
    if respect_existing_env:
        for name in _ENTRYPOINT_PLATFORM_ENV_VARS:
            env_value = _normalize_entrypoint_platform(os.environ.get(name))
            if env_value is not None:
                return env_value
    return _normalize_entrypoint_platform(default)


def apply_entrypoint_jax_runtime_env(platform: str | None) -> str | None:
    """Synchronize platform/x64 env vars before any entrypoint imports ``jax``."""
    normalized = _normalize_entrypoint_platform(platform)
    os.environ.setdefault(_JAX_ENABLE_X64_ENV, "True")
    for name in _ENTRYPOINT_PLATFORM_ENV_VARS:
        os.environ.pop(name, None)
    os.environ.pop(_XLA_PREALLOCATE_ENV, None)
    if normalized is None:
        return None
    for name in _ENTRYPOINT_PLATFORM_ENV_VARS:
        os.environ[name] = normalized
    if normalized == "cuda":
        os.environ.setdefault(_XLA_PREALLOCATE_ENV, "false")
    return normalized


def configure_entrypoint_jax_runtime(
    argv: list[str] | None = None,
    *,
    default_platform: str | None = None,
    respect_existing_env: bool = True,
) -> str | None:
    """Apply entrypoint JAX env policy before any direct ``import jax``."""
    requested_platform = preparse_entrypoint_jax_platform(
        [] if argv is None else argv,
        default=default_platform,
        respect_existing_env=respect_existing_env,
    )
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
    package_root = Path(src_root) / "simsopt"
    _strip_editable_finders()
    existing = sys.modules.get("simsopt")

    if _is_same_local_package(existing, package_root):
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
