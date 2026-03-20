"""Shared helpers for the JAX GPU validation ladder probes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def preparse_platform(argv: list[str]) -> str:
    """Read --platform before JAX import so scripts can pin the runtime device."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--platform", choices=("auto", "cpu", "cuda"), default="auto")
    args, _ = parser.parse_known_args(argv)
    return args.platform


def apply_requested_platform(platform: str) -> None:
    """Pin JAX to a specific platform before importing the package."""
    if platform != "auto":
        os.environ["JAX_PLATFORMS"] = platform


def repo_pythonpath_env(*, platform: str = "auto") -> dict[str, str]:
    """Return an environment that resolves in-repo imports for subprocess probes."""
    env = dict(os.environ)
    if platform != "auto":
        env["JAX_PLATFORMS"] = platform
    pythonpath_entries = [str(REPO_ROOT), str(SRC_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def bootstrap_local_simsopt() -> None:
    """Force imports to resolve against this repo's source tree."""
    package_root = SRC_ROOT / "simsopt"
    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if not (
            type(finder).__name__ == "ScikitBuildRedirectingFinder"
            and type(finder).__module__ == "_simsopt_editable"
        )
    ]
    for name in list(sys.modules):
        if name == "simsopt" or name.startswith("simsopt."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        "simsopt",
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to bootstrap local simsopt package from {package_root}")
    module = importlib.util.module_from_spec(spec)
    module.__path__ = [str(package_root)]
    sys.modules["simsopt"] = module
    spec.loader.exec_module(module)


def get_git_sha() -> str:
    """Return the exact repo SHA for provenance."""
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def relative_error(actual: float, reference: float) -> float:
    """Stable scalar relative error helper."""
    return float(abs(actual - reference) / (abs(reference) + 1e-30))


def max_relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    """Return the maximum elementwise relative error."""
    actual_arr = np.asarray(actual, dtype=float)
    reference_arr = np.asarray(reference, dtype=float)
    denom = np.maximum(np.abs(reference_arr), 1e-30)
    return float(np.max(np.abs(actual_arr - reference_arr) / denom))


def l2_relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    """Return the vector L2 relative error."""
    actual_arr = np.asarray(actual, dtype=float)
    reference_arr = np.asarray(reference, dtype=float)
    return float(
        np.linalg.norm(actual_arr - reference_arr)
        / (np.linalg.norm(reference_arr) + 1e-30)
    )


def peak_rss_mb() -> float:
    """Return the process max RSS in MB using platform-correct units."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


def query_gpu_memory_mb() -> float | None:
    """Return coarse GPU memory usage from nvidia-smi when available."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return float(lines[0])
    except ValueError:
        return None


def build_provenance(jax_module, jaxlib_module, *, title: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect shared provenance fields for ladder outputs."""
    provenance = {
        "title": title,
        "repo_sha": get_git_sha(),
        "jax": jax_module.__version__,
        "jaxlib": jaxlib_module.__version__,
        "backend": jax_module.default_backend(),
        "devices": [str(device) for device in jax_module.devices()],
        "x64_enabled": bool(jax_module.numpy.zeros(1).dtype == jax_module.numpy.float64),
        "peak_rss_mb": peak_rss_mb(),
    }
    gpu_memory_mb = query_gpu_memory_mb()
    if gpu_memory_mb is not None:
        provenance["gpu_memory_mb"] = gpu_memory_mb
    if extra:
        provenance.update(extra)
    return provenance


def print_provenance(provenance: dict[str, Any]) -> None:
    """Emit a stable human-readable provenance block."""
    print(f"\n{'=' * 70}")
    print(provenance["title"])
    print(f"{'=' * 70}")
    print(f"repo sha:     {provenance['repo_sha']}")
    print(f"jax:          {provenance['jax']}")
    print(f"jaxlib:       {provenance['jaxlib']}")
    print(f"backend:      {provenance['backend']}")
    print(f"devices:      {provenance['devices']}")
    print(f"x64 enabled:  {provenance['x64_enabled']}")
    if "fixture" in provenance:
        print(f"fixture:      {provenance['fixture']}")
    if "config_label" in provenance:
        print(f"config:       {provenance['config_label']}")
    if "platform_request" in provenance:
        print(f"platform arg: {provenance['platform_request']}")
    if "peak_rss_mb" in provenance:
        print(f"peak RSS:     {provenance['peak_rss_mb']:.1f} MB")
    if "gpu_memory_mb" in provenance:
        print(f"GPU memory:   {provenance['gpu_memory_mb']:.1f} MB")


def write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    """Write JSON payloads for probe outputs."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2)


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a JSON payload."""
    with Path(path).open("r", encoding="utf-8") as infile:
        return json.load(infile)


def run_python_script(
    script_path: str | os.PathLike[str],
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a Python helper script using the current interpreter."""
    command = [sys.executable, str(script_path), *args]
    return subprocess.run(
        command,
        cwd=str(cwd or REPO_ROOT),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def find_single_file(root: str | os.PathLike[str], pattern: str) -> Path:
    """Resolve a single probe artifact under a temporary output root."""
    matches = list(Path(root).rglob(pattern))
    if len(matches) != 1:
        match_display = ", ".join(str(match) for match in matches) or "<none>"
        raise RuntimeError(f"Expected exactly one {pattern!r} under {root}, found {match_display}")
    return matches[0]
