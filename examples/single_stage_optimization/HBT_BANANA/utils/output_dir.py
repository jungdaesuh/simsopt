"""Resolve the output directory for banana drivers.

Priority:
  1. BANANA_OUT_DIR env var (manual override)
  2. HBT_BANANA/outputs/
"""

import os

LOCAL_SUBDIR = "outputs"
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def resolve_output_dir():
    """Return the absolute path to the output directory, creating it if needed."""
    # 1. Explicit override
    env_dir = os.environ.get("BANANA_OUT_DIR")
    if env_dir:
        out = os.path.abspath(env_dir)
        os.makedirs(out, exist_ok=True)
        return out

    # 2. Local default
    out = os.path.join(PROJECT_DIR, LOCAL_SUBDIR)
    os.makedirs(out, exist_ok=True)
    return out
