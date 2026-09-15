"""Run a native-cpu child with OpenMP pinned before the extension loads."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from examples.jax.parity.runtime import build_parity_lane_environment


def run_native_cpu_child(
    source: str,
    *args: str,
    repo_root: Path,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute ``source`` as ``python -S -c`` under the native-cpu lane env.

    ``build_parity_lane_environment("native-cpu", ...)`` is the SSOT that
    sets ``OMP_NUM_THREADS=1`` before libgomp starts. An in-process env
    pin cannot undo the pytest process team.
    """
    environment = build_parity_lane_environment(
        "native-cpu", dict(os.environ), repo_root=repo_root
    )
    return subprocess.run(
        (sys.executable, "-S", "-c", source, *args),
        cwd=repo_root if cwd is None else cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
